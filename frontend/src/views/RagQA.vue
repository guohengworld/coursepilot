<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { getCourses, askQuestion } from '@/api/courses'
import { useSSE } from '@/composables/useSSE'
import type { Course, AskResponse } from '@/types'

const route = useRoute()
const auth = useAuthStore()

const courses = ref<Course[]>([])
const selectedCourseId = ref(route.query.courseId as string || '')
const question = ref('')
const answer = ref('')
const metadata = ref<AskResponse | null>(null)
const loading = ref(false)
const errorMsg = ref('')
const mode = ref<'stream' | 'sync'>('stream')

// 本地会话历史
interface QaSession {
  id: number
  question: string
  answer: string
  metadata: AskResponse | null
}
const qaHistory = ref<QaSession[]>([])
let sessionCounter = 0

const { isStreaming, streamAsk, stop } = useSSE()

async function loadCourses() {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
}

function handleNewSession() {
  // 保存当前会话到历史
  if (question.value || answer.value) {
    qaHistory.value.unshift({
      id: ++sessionCounter,
      question: question.value,
      answer: answer.value,
      metadata: metadata.value ? { ...metadata.value } : null,
    })
  }
  // 清空当前输入
  question.value = ''
  answer.value = ''
  metadata.value = null
  errorMsg.value = ''
}

function showHistoryItem(item: QaSession) {
  question.value = item.question
  answer.value = item.answer
  metadata.value = item.metadata
  errorMsg.value = ''
}

async function handleAsk() {
  if (!selectedCourseId.value || !question.value) return

  // 清空回答区
  answer.value = ''
  metadata.value = null
  errorMsg.value = ''

  if (mode.value === 'stream') {
    loading.value = true
    await streamAsk(
      selectedCourseId.value,
      question.value,
      () => auth.token,
      (token) => {
        answer.value += token
      },
      () => {
        loading.value = false
      },
      (err) => {
        errorMsg.value = err
        loading.value = false
      },
    )
  } else {
    loading.value = true
    const res = await askQuestion(selectedCourseId.value, { question: question.value })
    if (res.ok && !('detail' in res.data)) {
      const data = res.data as AskResponse
      answer.value = data.answer
      metadata.value = data
    } else {
      errorMsg.value = (res.data as any)?.detail || '请求失败'
    }
    loading.value = false
  }
}

function handleStop() {
  stop()
  loading.value = false
}

onMounted(loadCourses)
</script>

<template>
  <div class="page-container rag-qa-page">
    <div class="page-header">
      <h2>RAG 问答</h2>
      <el-button type="primary" @click="handleNewSession" :disabled="isStreaming">
        新建会话
      </el-button>
    </div>

    <div class="qa-layout">
      <!-- 主问答区 -->
      <div class="qa-main">
        <el-card shadow="never" class="mb-16">
          <el-form label-position="top">
            <el-form-item label="选择课程">
              <el-select
                v-model="selectedCourseId"
                placeholder="请选择课程"
                style="width: 320px"
              >
                <el-option
                  v-for="c in courses"
                  :key="c.id"
                  :label="c.name"
                  :value="c.id"
                />
              </el-select>
            </el-form-item>

            <el-form-item label="模式">
              <el-radio-group v-model="mode">
                <el-radio value="stream">流式（逐 token 展示）</el-radio>
                <el-radio value="sync">同步（完整回答）</el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item label="问题">
              <el-input
                v-model="question"
                type="textarea"
                :rows="3"
                placeholder="请输入关于课程内容的问题..."
                :disabled="isStreaming"
              />
            </el-form-item>

            <div class="action-row">
              <el-button
                v-if="!isStreaming"
                type="primary"
                :loading="loading && !isStreaming"
                :disabled="!selectedCourseId || !question"
                @click="handleAsk"
              >
                提问
              </el-button>
              <el-button v-else type="danger" @click="handleStop">停止</el-button>
            </div>
          </el-form>
        </el-card>

        <el-card v-if="answer" shadow="never">
          <template #header>
            <span>回答</span>
          </template>

          <div class="answer-content">{{ answer }}</div>

          <template v-if="isStreaming">
            <el-icon class="is-loading" :size="16"><Loading /></el-icon>
            <span class="text-secondary" style="margin-left: 6px;">生成中...</span>
          </template>

          <el-divider v-if="metadata" />

          <div v-if="metadata" class="metadata-section">
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item label="改写后查询">
                <code>{{ metadata.rewritten_query }}</code>
              </el-descriptions-item>
              <el-descriptions-item label="Trace ID">
                <code>{{ metadata.trace_id }}</code>
              </el-descriptions-item>
              <el-descriptions-item label="来源知识点">
                <el-tag
                  v-for="(kp, i) in metadata.source_kp_paths"
                  :key="i"
                  size="small"
                  style="margin: 2px"
                >
                  {{ kp }}
                </el-tag>
                <span v-if="!metadata.source_kp_paths.length" class="text-secondary">无</span>
              </el-descriptions-item>
              <el-descriptions-item label="得分">
                <span v-for="(score, i) in metadata.top_scores" :key="i" class="score-badge">
                  {{ (score * 100).toFixed(1) }}%
                </span>
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>

        <el-alert
          v-if="errorMsg"
          :title="errorMsg"
          type="error"
          show-icon
          :closable="true"
          @close="errorMsg = ''"
          class="mt-16"
        />
      </div>

      <!-- 历史会话侧边栏 -->
      <div class="qa-sidebar" v-if="qaHistory.length > 0">
        <el-card shadow="never">
          <template #header><span>历史会话</span></template>
          <div
            v-for="item in qaHistory"
            :key="item.id"
            class="history-item"
            @click="showHistoryItem(item)"
          >
            <div class="history-question">{{ item.question }}</div>
            <div class="history-preview text-secondary">{{ item.answer.slice(0, 80) }}{{ item.answer.length > 80 ? '...' : '' }}</div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rag-qa-page {
  height: calc(100vh - var(--header-height) - 48px);
  display: flex;
  flex-direction: column;
}

.qa-layout {
  display: flex;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

.qa-main {
  flex: 1;
  overflow-y: auto;
}

.qa-sidebar {
  width: 260px;
  flex-shrink: 0;
  overflow-y: auto;
}

.history-item {
  padding: 10px 8px;
  border-bottom: 1px solid var(--el-border-color-light);
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.2s;
  margin-bottom: 2px;
}

.history-item:hover {
  background: var(--el-color-primary-light-9);
}

.history-question {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 4px;
}

.history-preview {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.answer-content {
  white-space: pre-wrap;
  line-height: 1.7;
  font-size: 15px;
}

.action-row {
  display: flex;
  gap: 12px;
}

.metadata-section {
  margin-top: 8px;
}

.metadata-section code {
  font-size: 12px;
  word-break: break-all;
}

.score-badge {
  display: inline-block;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  margin: 2px;
}
</style>
