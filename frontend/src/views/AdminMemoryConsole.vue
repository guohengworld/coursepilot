<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { getMemoryDashboard, getSessionMemory, recallMemory } from '@/api/admin'
import type { MemoryDashboardResponse, SessionMemoryResponse, MemoryRecallResponse } from '@/api/admin'

const auth = useAuthStore()

const courseId = ref('')
const days = ref(7)
const dashboardLoading = ref(false)
const dashboard = ref<MemoryDashboardResponse | null>(null)

const sessionId = ref('')
const sessionLoading = ref(false)
const sessionDetail = ref<SessionMemoryResponse | null>(null)

const recallUserId = ref('')
const recallCourseId = ref('')
const recallQuery = ref('')
const recallLoading = ref(false)
const recallResult = ref<MemoryRecallResponse | null>(null)

async function loadDashboard() {
  if (!courseId.value) {
    ElMessage.warning('请输入课程 ID')
    return
  }
  dashboardLoading.value = true
  const res = await getMemoryDashboard(courseId.value, days.value)
  dashboardLoading.value = false
  if (res.ok) {
    dashboard.value = res.data as MemoryDashboardResponse
  } else {
    ElMessage.error((res.data as any)?.detail || '加载失败')
  }
}

async function loadSession() {
  if (!sessionId.value) {
    ElMessage.warning('请输入会话 ID')
    return
  }
  sessionLoading.value = true
  const res = await getSessionMemory(sessionId.value)
  sessionLoading.value = false
  if (res.ok) {
    sessionDetail.value = res.data as SessionMemoryResponse
  } else {
    ElMessage.error((res.data as any)?.detail || '加载失败')
  }
}

async function doRecall() {
  if (!recallUserId.value || !recallCourseId.value || !recallQuery.value) {
    ElMessage.warning('请填写完整召回参数')
    return
  }
  recallLoading.value = true
  const res = await recallMemory(recallUserId.value, recallCourseId.value, recallQuery.value)
  recallLoading.value = false
  if (res.ok) {
    recallResult.value = res.data as MemoryRecallResponse
  } else {
    ElMessage.error((res.data as any)?.detail || '召回失败')
  }
}

onMounted(() => {
  if (!auth.isSuperuser) {
    ElMessage.error('无权访问：仅限 admin')
  }
})
</script>

<template>
  <div class="page-container admin-memory-console">
    <div class="page-header">
      <h2>记忆层控制台（Admin）</h2>
      <el-tag v-if="auth.isSuperuser" type="success">super</el-tag>
      <el-tag v-else type="danger">无权限</el-tag>
    </div>

    <el-alert
      v-if="!auth.isSuperuser"
      title="当前账号无权限访问此页面"
      type="error"
      :closable="false"
      class="mb-16"
    />

    <!-- Dashboard -->
    <el-card shadow="never" class="mb-16">
      <template #header>
        <span>课程记忆层仪表盘</span>
      </template>
      <div class="form-row">
        <el-input v-model="courseId" placeholder="课程 ID" clearable style="flex: 1;" />
        <el-input-number v-model="days" :min="1" :max="90" style="width: 120px;" />
        <el-button type="primary" :loading="dashboardLoading" @click="loadDashboard">查询</el-button>
      </div>

      <div v-if="dashboard" class="dashboard-content">
        <el-row :gutter="16" class="mt-16">
          <el-col :span="8">
            <el-statistic title="总会话数" :value="dashboard.course_stats.total_sessions" />
          </el-col>
          <el-col :span="8">
            <el-statistic title="总 Token" :value="dashboard.course_stats.total_tokens" />
          </el-col>
          <el-col :span="8">
            <el-statistic title="总成本(元)" :value="dashboard.course_stats.total_cost" :precision="4" />
          </el-col>
        </el-row>

        <el-row :gutter="16" class="mt-16">
          <el-col :span="8">
            <el-statistic
              title="QA embedding 覆盖率"
              :value="dashboard.memory_layer_stats.embedding_coverage * 100"
              suffix="%"
              :precision="0"
            />
          </el-col>
          <el-col :span="8">
            <el-statistic
              title="Profile facts 覆盖率"
              :value="dashboard.memory_layer_stats.facts_coverage * 100"
              suffix="%"
              :precision="0"
            />
          </el-col>
          <el-col :span="8">
            <el-statistic title="QA 总数" :value="dashboard.memory_layer_stats.qa_total" />
          </el-col>
        </el-row>

        <h4 class="mt-16">最近会话</h4>
        <el-table :data="dashboard.recent_sessions" size="small" max-height="400">
          <el-table-column prop="session_id" label="会话 ID" width="220" />
          <el-table-column prop="intent" label="意图" width="100" />
          <el-table-column prop="status" label="状态" width="100" />
          <el-table-column prop="token_count" label="Token" width="100" />
          <el-table-column prop="conversation_turns" label="L1 轮次" width="100" />
          <el-table-column prop="has_rolling_summary" label="含 L2" width="80">
            <template #default="{ row }">
              <el-tag :type="row.has_rolling_summary ? 'success' : 'info'" size="small">
                {{ row.has_rolling_summary ? '是' : '否' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" label="创建时间" />
        </el-table>
      </div>
    </el-card>

    <!-- Session Detail -->
    <el-card shadow="never" class="mb-16">
      <template #header>
        <span>会话记忆详情</span>
      </template>
      <div class="form-row">
        <el-input v-model="sessionId" placeholder="会话 ID" clearable style="flex: 1;" />
        <el-button type="primary" :loading="sessionLoading" @click="loadSession">查询</el-button>
      </div>

      <div v-if="sessionDetail" class="session-content">
        <el-descriptions :column="3" border size="small" class="mt-16">
          <el-descriptions-item label="会话 ID">{{ sessionDetail.session_id }}</el-descriptions-item>
          <el-descriptions-item label="意图">{{ sessionDetail.intent }}</el-descriptions-item>
          <el-descriptions-item label="状态">{{ sessionDetail.status }}</el-descriptions-item>
          <el-descriptions-item label="L1 轮次">{{ sessionDetail.context_metrics.l1_turns }}</el-descriptions-item>
          <el-descriptions-item label="L1 tokens">{{ sessionDetail.context_metrics.l1_tokens }}</el-descriptions-item>
          <el-descriptions-item label="L2 tokens">{{ sessionDetail.context_metrics.l2_tokens }}</el-descriptions-item>
        </el-descriptions>

        <h4 class="mt-16">L1 近期对话</h4>
        <el-timeline v-if="sessionDetail.conversation && sessionDetail.conversation.length">
          <el-timeline-item
            v-for="(turn, idx) in sessionDetail.conversation"
            :key="idx"
            :type="turn.role === 'user' ? 'primary' : 'success'"
          >
            <strong>{{ turn.role }}:</strong> {{ turn.content }}
          </el-timeline-item>
        </el-timeline>
        <el-empty v-else description="无对话记录" />

        <h4 class="mt-16">L2 滚动摘要</h4>
        <pre v-if="sessionDetail.rolling_summary" class="code-block">{{ sessionDetail.rolling_summary }}</pre>
        <el-empty v-else description="无滚动摘要" />

        <h4 class="mt-16">L3 语义记忆</h4>
        <pre v-if="sessionDetail.memory_facts" class="code-block">{{ JSON.stringify(sessionDetail.memory_facts, null, 2) }}</pre>
        <el-empty v-else description="无语义记忆" />
      </div>
    </el-card>

    <!-- Memory Recall -->
    <el-card shadow="never">
      <template #header>
        <span>L4 归档记忆召回测试</span>
      </template>
      <div class="form-row">
        <el-input v-model="recallUserId" placeholder="用户 ID" clearable style="flex: 1;" />
        <el-input v-model="recallCourseId" placeholder="课程 ID" clearable style="flex: 1;" />
        <el-input v-model="recallQuery" placeholder="查询文本" clearable style="flex: 2;" />
        <el-button type="primary" :loading="recallLoading" @click="doRecall">召回</el-button>
      </div>

      <div v-if="recallResult" class="recall-content">
        <h4 class="mt-16">召回结果（{{ recallResult.results.length }} 条）</h4>
        <el-card v-for="(item, idx) in recallResult.results" :key="idx" shadow="never" class="mb-8">
          <div><strong>Query:</strong> {{ item.query }}</div>
          <div><strong>Answer:</strong> {{ item.answer }}</div>
          <div class="mt-8">
            <el-tag size="small">score: {{ item.scores.score }}</el-tag>
            <el-tag size="small" type="info">recency: {{ item.scores.recency }}</el-tag>
            <el-tag size="small" type="warning">relevance: {{ item.scores.relevance }}</el-tag>
            <el-tag size="small" type="success">importance: {{ item.scores.importance }}</el-tag>
          </div>
        </el-card>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.admin-memory-console {
  max-width: 1200px;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.form-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

.mt-16 {
  margin-top: 16px;
}

.mb-8 {
  margin-bottom: 8px;
}

.mb-16 {
  margin-bottom: 16px;
}

.code-block {
  background: var(--el-fill-color);
  padding: 12px;
  border-radius: 4px;
  max-height: 300px;
  overflow: auto;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
