<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import {
  getCourse,
  getDocuments,
  deleteDocument,
  uploadDocument,
  enrollCourse,
} from '@/api/courses'
import type { Course, Document } from '@/types'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const courseId = route.params.id as string

const course = ref<Course | null>(null)
const documents = ref<Document[]>([])
const loading = ref(true)
const docLoading = ref(false)
// 403 归属拦截态：展示"加入课程"面板，不直接踢回列表
const denied = ref(false)
const joining = ref(false)

// Upload
const uploadVisible = ref(false)
const uploadFile = ref<File | null>(null)
const uploading = ref(false)

async function loadCourse() {
  const res = await getCourse(courseId)
  if (res.ok && !('detail' in res.data)) {
    course.value = res.data as Course
  } else {
    // ② 归属校验：非本课程成员一律 403（后端不区分"不存在/无权"，避免泄露课程存在性）
    if (res.status === 403) {
      denied.value = true
      ElMessage.warning('您不是该课程的成员，加入课程后即可查看资料并使用问答')
    } else {
      ElMessage.error('课程不存在或已删除')
      router.push('/courses')
    }
    return
  }
}

async function handleEnroll() {
  joining.value = true
  const res = await enrollCourse(courseId)
  if (res.ok) {
    ElMessage.success('加入成功，正在进入课程...')
    denied.value = false
    await loadCourse()
    if (course.value) {
      await loadDocuments()
    }
  } else {
    ElMessage.error((res.data as any)?.detail || '加入失败')
  }
  joining.value = false
}

async function loadDocuments() {
  docLoading.value = true
  const res = await getDocuments(courseId)
  if (res.ok && Array.isArray(res.data)) {
    documents.value = res.data as Document[]
  } else if (res.status === 403) {
    documents.value = []
    ElMessage.warning('您不是该课程的成员，无法查看课程资料')
  }
  docLoading.value = false
}

async function handleUpload() {
  if (!uploadFile.value) return
  uploading.value = true
  const res = await uploadDocument(courseId, uploadFile.value)
  if (res.ok) {
    ElMessage.success('文档上传成功，正在处理中...')
    uploadVisible.value = false
    uploadFile.value = null
    await loadDocuments()
  } else {
    ElMessage.error((res.data as any)?.detail || '上传失败')
  }
  uploading.value = false
}

async function handleDeleteDoc(docId: string, filename: string) {
  try {
    await ElMessageBox.confirm(`确定要删除文档「${filename}」吗？`, '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    const res = await deleteDocument(courseId, docId)
    if (res.ok) {
      ElMessage.success('文档已删除')
      await loadDocuments()
    } else {
      ElMessage.error((res.data as any)?.detail || '删除失败')
    }
  } catch {
    // cancelled
  }
}

onMounted(async () => {
  loading.value = true
  await loadCourse()
  if (!denied.value && course.value) {
    await loadDocuments()
  }
  loading.value = false
})

function fileTypeTag(type: string) {
  const map: Record<string, string> = { pdf: 'danger', docx: 'primary', md: 'success' }
  return map[type] || 'info'
}

function statusTag(status: string) {
  const map: Record<string, string> = { ready: 'success', processing: 'warning', failed: 'danger', pending: 'info' }
  return map[status] || 'info'
}

const fileAccept = '.pdf,.docx,.md'

function onUploadChange(file: any) {
  uploadFile.value = file.raw
}
</script>

<template>
  <div class="page-container" v-loading="loading">
    <div class="page-header">
      <h2>{{ course?.name || '课程详情' }}</h2>
      <el-button @click="router.push('/courses')">
        <el-icon><Back /></el-icon> 返回列表
      </el-button>
    </div>

    <el-result
      v-if="denied && !course"
      icon="warning"
      title="您不是该课程的成员"
      sub-title="加入课程后即可查看课程资料并使用 AI 问答"
    >
      <template #extra>
        <el-space>
          <el-button type="primary" :loading="joining" @click="handleEnroll">
            {{ joining ? '加入中...' : '加入课程' }}
          </el-button>
          <el-button @click="router.push('/courses')">返回课程列表</el-button>
        </el-space>
      </template>
    </el-result>

    <el-card v-if="course && !denied" class="mb-16" shadow="never">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="课程名称">{{ course.name }}</el-descriptions-item>
        <el-descriptions-item label="描述">{{ course.description || '-' }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ new Date(course.created_at).toLocaleString('zh-CN') }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- Quick actions -->
    <el-card v-if="course && !denied" class="mb-16" shadow="never">
      <template #header><span>快捷操作</span></template>
      <el-space wrap>
        <el-button @click="router.push(`/knowledge-points?courseId=${courseId}`)">
          <el-icon><Share /></el-icon> 知识点树
        </el-button>
        <el-button @click="router.push(`/rag-qa?courseId=${courseId}`)">
          <el-icon><ChatLineSquare /></el-icon> RAG 问答
        </el-button>
        <el-button @click="router.push(`/agent?courseId=${courseId}`)">
          <el-icon><MagicStick /></el-icon> Agent 对话
        </el-button>
        <el-button
          v-if="auth.isTeacher"
          type="primary"
          plain
          @click="router.push(`/tasks?courseId=${courseId}`)"
        >
          <el-icon><Tickets /></el-icon> 布置任务
        </el-button>
      </el-space>
    </el-card>

    <!-- Documents -->
    <el-card v-if="course && !denied" shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>文档管理</span>
          <el-button
            v-if="auth.isTeacher"
            type="primary"
            size="small"
            @click="uploadVisible = true"
          >
            <el-icon><Upload /></el-icon> 上传文档
          </el-button>
        </div>
      </template>

      <el-table :data="documents" v-loading="docLoading" stripe style="width: 100%">
        <el-table-column prop="filename" label="文件名" min-width="200" />
        <el-table-column prop="file_type" label="类型" width="80">
          <template #default="{ row }">
            <el-tag :type="fileTypeTag(row.file_type)" size="small">{{ row.file_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="file_size" label="大小" width="100">
          <template #default="{ row }">
            {{ row.file_size ? (row.file_size / 1024).toFixed(1) + ' KB' : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="上传时间" width="160">
          <template #default="{ row }">
            {{ new Date(row.uploaded_at).toLocaleString('zh-CN') }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="auth.isSuperuser"
              size="small"
              type="danger"
              @click="handleDeleteDoc(row.id, row.filename)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-if="!docLoading && documents.length === 0" description="暂无文档" />
    </el-card>

    <!-- Upload Dialog -->
    <el-dialog v-model="uploadVisible" title="上传文档" width="500px">
      <el-upload
        drag
        :auto-upload="false"
        :show-file-list="true"
        :accept="fileAccept"
        :on-change="onUploadChange"
        :limit="1"
      >
        <el-icon class="el-icon--upload" :size="48"><UploadFilled /></el-icon>
        <div class="el-upload__text">将文件拖到此处，或<em>点击选择</em></div>
        <template #tip>
          <div class="el-upload__tip">支持 PDF、DOCX、MD 格式</div>
        </template>
      </el-upload>
      <template #footer>
        <el-button @click="uploadVisible = false">取消</el-button>
        <el-button type="primary" :loading="uploading" :disabled="!uploadFile" @click="handleUpload">
          {{ uploading ? '上传中...' : '上传' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

