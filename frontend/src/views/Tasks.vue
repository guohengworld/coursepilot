<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { getCourses, getKnowledgePoints } from '@/api/courses'
import {
  createTaskDraft,
  getTask,
  getTaskCandidates,
  getTasks,
  publishTask,
  updateTask,
} from '@/api/tasks'
import type {
  Course,
  KnowledgePoint,
  TaskAcceptance,
  TaskDetail,
  TaskGoal,
  TaskGroup,
  TaskListItem,
} from '@/types'

const auth = useAuthStore()
const route = useRoute()

const isTeacherView = computed(() => auth.isTeacher)

const fmtTime = (s: string | null | undefined) =>
  s ? new Date(s).toLocaleString('zh-CN') : '-'

const DIFF_LABELS = ['', '易', '较易', '中等', '较难', '难']

// ===== 课程 / 学生（教师创建侧） =====
const courses = ref<Course[]>([])
const coursesLoaded = ref(false)
const courseOptions = computed(() =>
  courses.value.filter((c) => auth.isSuperuser || c.created_by === auth.user?.id),
)
const courseNameMap = computed(() => {
  const m = new Map<string, string>()
  for (const c of courses.value) m.set(c.id, c.name)
  return m
})
const selectedCourseId = ref((route.query.courseId as string) || '')
const candidates = ref<{ user_id: string; username: string }[]>([])
const selectedStudentId = ref('')
const kpOptions = ref<KnowledgePoint[]>([])
const generating = ref(false)

/** 会话内学生名缓存：列表接口不返回 username，本会话生成过的能显示真名 */
const studentNameCache = new Map<string, string>()
function studentLabel(task: { student_id: string }) {
  return studentNameCache.get(task.student_id) || `学生 ${task.student_id.slice(0, 8)}`
}

async function loadCourses() {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
  coursesLoaded.value = true
}

async function loadCandidates() {
  candidates.value = []
  selectedStudentId.value = ''
  if (!selectedCourseId.value || !isTeacherView.value) return
  const res = await getTaskCandidates(selectedCourseId.value)
  if (res.ok && Array.isArray(res.data)) {
    candidates.value = res.data as { user_id: string; username: string }[]
    if (!candidates.value.length) {
      ElMessage.warning('该课程暂无学生（enrollments 尚未回填）')
    }
  } else {
    ElMessage.warning((res.data as { detail?: string }).detail || '获取学生名单失败')
  }
}

async function loadKpOptions() {
  kpOptions.value = []
  if (!selectedCourseId.value || !isTeacherView.value) return
  const res = await getKnowledgePoints(selectedCourseId.value)
  if (res.ok && Array.isArray(res.data)) {
    kpOptions.value = res.data as KnowledgePoint[]
  }
}

watch(selectedCourseId, () => {
  loadCandidates()
  loadKpOptions()
})

// ===== 任务列表 =====
const tasks = ref<TaskListItem[]>([])
const listLoading = ref(false)
const statusFilter = ref<'all' | 'draft' | 'published'>('all')

const filteredTasks = computed(() =>
  statusFilter.value === 'all'
    ? tasks.value
    : tasks.value.filter((t) => t.status === statusFilter.value),
)

async function loadTasks() {
  listLoading.value = true
  const res = await getTasks()
  if (res.ok && Array.isArray(res.data)) {
    tasks.value = res.data as TaskListItem[]
  } else {
    ElMessage.error((res.data as { detail?: string }).detail || '加载任务列表失败')
  }
  listLoading.value = false
}

// ===== 创建草稿 =====
async function handleGenerate() {
  if (!selectedCourseId.value) {
    ElMessage.warning('请先选择课程')
    return
  }
  if (!selectedStudentId.value) {
    ElMessage.warning('请先选择学生')
    return
  }
  generating.value = true
  const res = await createTaskDraft({
    course_id: selectedCourseId.value,
    student_id: selectedStudentId.value,
  })
  if (res.ok && !('detail' in res.data)) {
    const detail = res.data as TaskDetail
    const username = candidates.value.find((c) => c.user_id === detail.student_id)?.username
    if (username) studentNameCache.set(detail.student_id, username)
    ElMessage.success('任务草稿已生成')
    await loadTasks()
    openEditor(detail.id)
  } else {
    ElMessage.error((res.data as { detail?: string }).detail || '生成失败，请重试')
  }
  generating.value = false
}

// ===== 详情 / 审核 =====
const editorVisible = ref(false)
const editorReadonly = ref(true)
const currentId = ref('')
const saving = ref(false)
const publishing = ref(false)

const editor = reactive({
  courseId: '',
  studentId: '',
  status: 'draft' as TaskListItem['status'],
  diagnosis: {} as TaskDetail['diagnosis'],
  metric: '',
  description: '',
  groups: [] as TaskGroup[],
  timeLimit: null as number | null,
  passCondition: '',
  fallbackAction: '',
  createdAt: '',
  publishedAt: null as string | null,
})

const noLimit = ref(true)
watch(noLimit, (v) => {
  if (!v && !editor.timeLimit) editor.timeLimit = 60
})
watch(
  () => editor.timeLimit,
  (v) => {
    noLimit.value = !v
  },
)

const drawerTitle = computed(() => {
  const course = courseNameMap.value.get(editor.courseId) || ''
  const label = studentLabel({ student_id: editor.studentId })
  const kind = editorReadonly.value ? '任务详情' : '草稿审核'
  return `${course} · ${label} · ${kind}`
})

const totalCount = computed(() =>
  editor.groups.reduce((s, g) => s + (Number(g.question_count) || 0), 0),
)

const weakKps = computed(() => (editor.diagnosis?.weak_kps as string[] | undefined) ?? [])
const commonMistakes = computed(() => {
  const arr = (editor.diagnosis?.common_mistakes as unknown[] | undefined) ?? []
  return arr.map((x) => (typeof x === 'string' ? x : JSON.stringify(x)))
})
const avgRateText = computed(() => {
  const v = editor.diagnosis?.avg_correct_rate
  return typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : null
})
const masteryCount = computed(
  () =>
    Object.keys(
      (editor.diagnosis?.mastery_level as Record<string, unknown> | undefined) ?? {},
    ).length,
)

function applyDetail(d: TaskDetail) {
  const goal = (d.goal ?? {}) as TaskGoal
  const acc = (d.acceptance ?? {}) as TaskAcceptance
  editor.courseId = d.course_id
  editor.studentId = d.student_id
  editor.status = d.status
  editor.diagnosis = d.diagnosis ?? {}
  editor.metric = String(goal.metric ?? '')
  editor.description = String(goal.description ?? '')
  editor.groups = (d.groups ?? []).map((g) => ({ ...g }))
  editor.timeLimit = d.time_limit_minutes
  editor.passCondition = String(acc.pass_condition ?? '')
  editor.fallbackAction = String(acc.fallback_action ?? '')
  editor.createdAt = d.created_at
  editor.publishedAt = d.published_at
}

async function openEditor(id: string) {
  const res = await getTask(id)
  if (res.ok && !('detail' in res.data)) {
    const detail = res.data as TaskDetail
    applyDetail(detail)
    currentId.value = id
    editorReadonly.value = !isTeacherView.value || detail.status !== 'draft'
    editorVisible.value = true
  } else {
    ElMessage.error((res.data as { detail?: string }).detail || '任务不存在')
  }
}

function validateBeforeSave(): boolean {
  if (!editor.description.trim()) {
    ElMessage.warning('请填写任务目标描述')
    return false
  }
  if (!editor.passCondition.trim()) {
    ElMessage.warning('请填写验收标准（pass_condition）')
    return false
  }
  if (!editor.groups.length) {
    ElMessage.warning('至少保留一个题组')
    return false
  }
  for (const g of editor.groups) {
    const q = Number(g.question_count)
    if (!Number.isInteger(q) || q < 1 || q > 50) {
      ElMessage.warning('题组题量需为 1~50 的整数')
      return false
    }
  }
  return true
}

async function handleSave() {
  if (!validateBeforeSave()) return
  saving.value = true
  const res = await updateTask(currentId.value, {
    goal: {
      metric: editor.metric.trim() || 'practice_correct_rate',
      description: editor.description.trim(),
    },
    groups: editor.groups.map((g) => ({
      kp_path: g.kp_path,
      kp_name: g.kp_name || g.kp_path.split('/').pop() || g.kp_path,
      question_count: Math.round(Number(g.question_count)),
      difficulty: Math.round(Number(g.difficulty) || 3),
      source: g.source || null,
      reason: g.reason || null,
    })),
    time_limit_minutes: editor.timeLimit,
    acceptance: {
      pass_condition: editor.passCondition.trim(),
      fallback_action: editor.fallbackAction.trim() || null,
    },
  })
  if (res.ok && !('detail' in res.data)) {
    applyDetail(res.data as TaskDetail)
    ElMessage.success('草稿已保存')
    await loadTasks()
  } else {
    ElMessage.error((res.data as { detail?: string }).detail || '保存失败')
  }
  saving.value = false
}

async function publishCurrent(extraHint?: string) {
  try {
    await ElMessageBox.confirm(
      `${extraHint || ''}发布后学生将立即看到该任务，且草稿不可再编辑。确认发布？`,
      '发布任务',
      { confirmButtonText: '发布', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return
  }
  publishing.value = true
  const res = await publishTask(currentId.value)
  if (res.ok && !('detail' in res.data)) {
    applyDetail(res.data as TaskDetail)
    editorReadonly.value = true
    ElMessage.success('任务已发布')
    await loadTasks()
  } else {
    ElMessage.error((res.data as { detail?: string }).detail || '发布失败')
  }
  publishing.value = false
}

async function handlePublishFromList(id: string) {
  currentId.value = id
  await publishCurrent()
}

// ===== 添加 / 删除题组 =====
const addGroupVisible = ref(false)
const newGroup = reactive({ kpPath: '', questionCount: 5, difficulty: 3, reason: '' })

function handleAddGroup() {
  const kp = kpOptions.value.find((k) => k.kp_path === newGroup.kpPath)
  if (!kp) {
    ElMessage.warning('请从课程知识点中选择')
    return
  }
  editor.groups.push({
    kp_path: kp.kp_path,
    kp_name: kp.title || kp.kp_path.split('/').pop() || kp.kp_path,
    question_count: newGroup.questionCount,
    difficulty: newGroup.difficulty,
    source: '课程资料',
    reason: newGroup.reason.trim() || null,
  })
  addGroupVisible.value = false
  newGroup.kpPath = ''
  newGroup.reason = ''
  newGroup.questionCount = 5
  newGroup.difficulty = 3
}

function handleRemoveGroup(index: number) {
  editor.groups.splice(index, 1)
}

onMounted(async () => {
  await loadCourses()
  await loadTasks()
  if (selectedCourseId.value) {
    loadCandidates()
    loadKpOptions()
  }
})
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2>任务中心</h2>
    </div>

    <!-- 教师：发布任务 -->
    <el-card v-if="isTeacherView" class="mb-16" shadow="never">
      <template #header>
        <div class="card-head">
          <span>发布任务</span>
          <span class="text-secondary">AI 依据学情生成草稿 → 教师审核微调 → 发布给学生</span>
        </div>
      </template>

      <template v-if="coursesLoaded && !courseOptions.length">
        <el-empty description="您还不是任何课程的教师，无法布置任务" :image-size="80" />
      </template>

      <template v-else>
        <div class="create-row">
          <el-select
            v-model="selectedCourseId"
            filterable
            placeholder="1. 选择课程"
            style="width: 240px"
          >
            <el-option v-for="c in courseOptions" :key="c.id" :label="c.name" :value="c.id" />
          </el-select>
          <el-select
            v-model="selectedStudentId"
            filterable
            placeholder="2. 选择学生"
            style="width: 220px"
            :disabled="!candidates.length"
          >
            <el-option
              v-for="s in candidates"
              :key="s.user_id"
              :label="s.username"
              :value="s.user_id"
            />
          </el-select>
          <el-button
            type="primary"
            :loading="generating"
            :disabled="!selectedCourseId || !selectedStudentId"
            @click="handleGenerate"
          >
            {{ generating ? 'AI 生成中…' : '3. 生成草稿' }}
          </el-button>
        </div>
        <div
          v-if="selectedCourseId && candidates.length === 0 && !generating"
          class="text-secondary"
          style="margin-top: 8px"
        >
          ⚠ 该课程暂无可选学生——学生名单来自 enrollments，需先回填 / 学生先加入课程
        </div>
      </template>
    </el-card>

    <!-- 任务列表 -->
    <el-card shadow="never">
      <template #header>
        <div class="card-head">
          <span>{{ isTeacherView ? '我的任务' : '我收到的任务' }}</span>
          <el-radio-group v-if="isTeacherView" v-model="statusFilter" size="small">
            <el-radio-button value="all">全部</el-radio-button>
            <el-radio-button value="draft">草稿</el-radio-button>
            <el-radio-button value="published">已发布</el-radio-button>
          </el-radio-group>
        </div>
      </template>

      <el-table :data="filteredTasks" v-loading="listLoading" stripe>
        <el-table-column label="任务目标" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">
            {{ (row.goal && (row.goal as TaskGoal).description) || '-' }}
          </template>
        </el-table-column>
        <el-table-column v-if="isTeacherView" label="学生" width="130">
          <template #default="{ row }">{{ studentLabel(row) }}</template>
        </el-table-column>
        <el-table-column label="课程" min-width="140" show-overflow-tooltip>
          <template #default="{ row }">
            {{ courseNameMap.get(row.course_id) || '未知课程' }}
          </template>
        </el-table-column>
        <el-table-column label="题量" width="70">
          <template #default="{ row }">{{ row.total_count }}</template>
        </el-table-column>
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag
              :type="row.status === 'published' ? 'success' : 'warning'"
              size="small"
            >
              {{ row.status === 'published' ? '已发布' : '草稿' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="更新时间" width="155">
          <template #default="{ row }">{{ fmtTime(row.updated_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="openEditor(row.id)">查看</el-button>
            <el-button
              v-if="isTeacherView && row.status === 'draft'"
              size="small"
              type="primary"
              @click="handlePublishFromList(row.id)"
            >
              发布
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!listLoading && !filteredTasks.length" description="暂无任务" />
    </el-card>

    <!-- 详情 / 审核抽屉 -->
    <el-drawer v-model="editorVisible" :size="760" :title="drawerTitle" destroy-on-close>
      <!-- ① 诊断依据（只读） -->
      <el-card v-if="Object.keys(editor.diagnosis).length" class="mb-16" shadow="never">
        <template #header><span>① 诊断依据（不可编辑，AI 生成任务的依据）</span></template>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="平均正确率">
            {{ avgRateText || '暂无练习数据' }}
          </el-descriptions-item>
          <el-descriptions-item label="班级位置">
            {{ editor.diagnosis.class_rank || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="薄弱知识点">
            <span v-if="weakKps.length">
              <el-tag
                v-for="w in weakKps"
                :key="w"
                size="small"
                type="danger"
                effect="plain"
                style="margin: 2px 4px 2px 0"
              >
                {{ w }}
              </el-tag>
            </span>
            <span v-else>-</span>
          </el-descriptions-item>
          <el-descriptions-item label="掌握度覆盖">
            {{ masteryCount ? `${masteryCount} 个知识点有画像` : '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="常见错误" :span="2">
            <ul v-if="commonMistakes.length" class="mistake-list">
              <li v-for="(m, i) in commonMistakes" :key="i">{{ m }}</li>
            </ul>
            <span v-else>-</span>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <!-- ② 任务目标 -->
      <el-card class="mb-16" shadow="never">
        <template #header><span>② 任务目标</span></template>
        <el-form label-position="top">
          <el-form-item label="目标指标 metric">
            <el-input
              v-model="editor.metric"
              :disabled="editorReadonly"
              placeholder="practice_correct_rate"
            />
          </el-form-item>
          <el-form-item label="目标描述">
            <el-input
              v-model="editor.description"
              type="textarea"
              :rows="2"
              :disabled="editorReadonly"
              placeholder="如：两周内将「二叉树遍历」正确率从 40% 提升到 75%"
            />
          </el-form-item>
        </el-form>
      </el-card>

      <!-- ③ 题组 -->
      <el-card class="mb-16" shadow="never">
        <template #header>
          <div class="card-head">
            <span>③ 任务题组（共 {{ totalCount }} 题）</span>
            <el-button
              v-if="!editorReadonly"
              size="small"
              type="primary"
              plain
              @click="addGroupVisible = true"
            >
              添加题组
            </el-button>
          </div>
        </template>

        <el-table :data="editor.groups" size="small" border>
          <el-table-column label="知识点" min-width="150">
            <template #default="{ row }">
              <div class="group-kp-name">{{ row.kp_name }}</div>
              <div class="group-kp-path text-secondary">{{ row.kp_path }}</div>
            </template>
          </el-table-column>
          <el-table-column label="题量" width="110">
            <template #default="{ row }">
              <el-input-number
                v-if="!editorReadonly"
                v-model="row.question_count"
                :min="1"
                :max="50"
                :step="1"
                size="small"
                controls-position="right"
                style="width: 100px"
              />
              <span v-else>{{ row.question_count }}</span>
            </template>
          </el-table-column>
          <el-table-column label="难度" width="90">
            <template #default="{ row }">
              <el-select
                v-if="!editorReadonly"
                v-model="row.difficulty"
                size="small"
                style="width: 72px"
              >
                <el-option
                  v-for="d in [1, 2, 3, 4, 5]"
                  :key="d"
                  :label="`${d}·${DIFF_LABELS[d]}`"
                  :value="d"
                />
              </el-select>
              <el-tag v-else size="small">
                {{ row.difficulty }}·{{ DIFF_LABELS[row.difficulty] || '' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="布置理由" min-width="150">
            <template #default="{ row }">
              <el-input
                v-if="!editorReadonly"
                v-model="row.reason"
                size="small"
                placeholder="布置理由"
              />
              <span v-else class="text-secondary">{{ row.reason || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column v-if="!editorReadonly" label="" width="60" align="center">
            <template #default="{ $index }">
              <el-button
                size="small"
                type="danger"
                text
                @click="handleRemoveGroup($index)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <div v-if="!editor.groups.length" class="text-secondary" style="padding: 8px">
          暂无题组
        </div>
      </el-card>

      <!-- ④ 验收标准 -->
      <el-card class="mb-16" shadow="never">
        <template #header><span>④ 验收标准与时限</span></template>
        <el-form label-position="top">
          <el-form-item label="完成时限">
            <el-checkbox v-if="!editorReadonly" v-model="noLimit" style="margin-right: 12px">
              不限时
            </el-checkbox>
            <span v-if="!editorReadonly && !noLimit">
              <el-input-number
                v-model="editor.timeLimit"
                :min="5"
                :max="600"
                :step="5"
                size="small"
                controls-position="right"
                style="width: 140px"
              />
              <span class="text-secondary" style="margin-left: 6px">分钟</span>
            </span>
            <span v-if="editorReadonly">
              {{ editor.timeLimit ? `${editor.timeLimit} 分钟` : '不限时' }}
            </span>
          </el-form-item>
          <el-form-item label="达标条件 pass_condition">
            <el-input
              v-model="editor.passCondition"
              type="textarea"
              :rows="2"
              :disabled="editorReadonly"
              placeholder="如：题组平均正确率 ≥ 70%"
            />
          </el-form-item>
          <el-form-item label="未达标后续动作 fallback_action（可选）">
            <el-input
              v-model="editor.fallbackAction"
              type="textarea"
              :rows="2"
              :disabled="editorReadonly"
              placeholder="如：系统追加一轮同知识点巩固练习"
            />
          </el-form-item>
        </el-form>
      </el-card>

      <div class="drawer-meta text-secondary">
        创建于 {{ fmtTime(editor.createdAt) }}
        <template v-if="editor.publishedAt">
          · 发布于 {{ fmtTime(editor.publishedAt) }}
        </template>
      </div>

      <template #footer>
        <el-button @click="editorVisible = false">关闭</el-button>
        <template v-if="!editorReadonly">
          <el-button :loading="saving" type="primary" @click="handleSave">
            保存修改
          </el-button>
          <el-button :loading="publishing" type="success" @click="publishCurrent()">
            发布
          </el-button>
        </template>
      </template>
    </el-drawer>

    <!-- 添加题组 -->
    <el-dialog v-model="addGroupVisible" title="添加题组" width="480px">
      <el-form label-position="top">
        <el-form-item label="知识点（取自该课程知识点树）" required>
          <el-select
            v-model="newGroup.kpPath"
            filterable
            placeholder="搜索知识点"
            style="width: 100%"
          >
            <el-option
              v-for="kp in kpOptions"
              :key="kp.id"
              :label="`${kp.title}（${kp.kp_path}）`"
              :value="kp.kp_path"
            />
          </el-select>
        </el-form-item>
        <el-form-item v-if="!kpOptions.length" class="text-secondary" style="margin-top: -8px">
          该课程暂无知识点，无法按知识点添加题组
        </el-form-item>
        <div class="add-group-row">
          <el-form-item label="题量">
            <el-input-number
              v-model="newGroup.questionCount"
              :min="1"
              :max="50"
              :step="1"
              size="small"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="难度">
            <el-select v-model="newGroup.difficulty" size="small" style="width: 130px">
              <el-option
                v-for="d in [1, 2, 3, 4, 5]"
                :key="d"
                :label="`${d}·${DIFF_LABELS[d]}`"
                :value="d"
              />
            </el-select>
          </el-form-item>
        </div>
        <el-form-item label="布置理由（可选）">
          <el-input
            v-model="newGroup.reason"
            type="textarea"
            :rows="2"
            placeholder="如：该知识点正确率仅 45%"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="addGroupVisible = false">取消</el-button>
        <el-button
          type="primary"
          :disabled="!newGroup.kpPath || !kpOptions.length"
          @click="handleAddGroup"
        >
          添加
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.create-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.mistake-list {
  margin: 0;
  padding-left: 18px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}

.group-kp-name {
  font-weight: 500;
  font-size: 13px;
}

.group-kp-path {
  font-size: 12px;
}

.add-group-row {
  display: flex;
  gap: 16px;
}

.drawer-meta {
  text-align: right;
  font-size: 12px;
}
</style>
