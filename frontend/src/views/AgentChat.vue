<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getCourses, chat, getSession, approveSession } from '@/api'
import type { Course } from '@/types'
import type { ChatResponse, SessionStatus } from '@/types'

const route = useRoute()
const courses = ref<Course[]>([])
const selectedCourseId = ref(route.query.courseId as string || '')
const message = ref('')
const sending = ref(false)

interface Message {
  role: 'user' | 'assistant'
  content: string
  intent?: string
  sources?: Record<string, unknown>[]
  sessionId?: string
}

const messages = ref<Message[]>([])
const sessions = ref<SessionStatus[]>([])
const activeSessionId = ref<string | null>(null)

async function loadCourses() {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
}

async function handleSend() {
  if (!selectedCourseId.value || !message.value) return

  const userMsg = message.value
  messages.value.push({ role: 'user', content: userMsg })
  message.value = ''
  sending.value = true

  const res = await chat({ message: userMsg, course_id: selectedCourseId.value })
  if (res.ok && !('detail' in res.data)) {
    const data = res.data as ChatResponse
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      intent: data.intent,
      sources: data.sources,
      sessionId: data.session_id,
    })
    activeSessionId.value = data.session_id
  } else {
    messages.value.push({
      role: 'assistant',
      content: `错误: ${(res.data as any)?.detail || '请求失败'}`,
    })
  }
  sending.value = false

  // Refresh sessions
  if (activeSessionId.value) {
    const sr = await getSession(activeSessionId.value)
    if (sr.ok && !('detail' in sr.data)) {
      const existing = sessions.value.findIndex((s) => s.session_id === activeSessionId.value)
      const data = sr.data as SessionStatus
      if (existing >= 0) {
        sessions.value[existing] = data
      } else {
        sessions.value.push(data)
      }
    }
  }
}

function intentTagType(intent: string) {
  const map: Record<string, string> = {
    question: 'primary',
    practice: 'success',
    diagnose: 'warning',
    review: 'info',
    code_help: 'danger',
  }
  return map[intent] || 'info'
}

onMounted(loadCourses)
</script>

<template>
  <div class="page-container chat-page">
    <div class="page-header">
      <h2>Agent 对话</h2>
    </div>

    <div class="chat-layout">
      <!-- Main chat area -->
      <div class="chat-main">
        <el-card class="mb-16">
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
        </el-card>

        <el-card class="chat-messages-card" shadow="never">
          <div class="chat-messages" v-if="messages.length > 0">
            <div
              v-for="(msg, i) in messages"
              :key="i"
              :class="['message-bubble', msg.role === 'user' ? 'user' : 'assistant']"
            >
              <div class="message-header">
                <strong>{{ msg.role === 'user' ? '你' : 'Agent' }}</strong>
                <el-tag
                  v-if="msg.intent"
                  :type="intentTagType(msg.intent)"
                  size="small"
                  effect="plain"
                >
                  {{ msg.intent }}
                </el-tag>
              </div>
              <div class="message-content">{{ msg.content }}</div>
              <div v-if="msg.sources && msg.sources.length" class="message-sources">
                <el-collapse>
                  <el-collapse-item title="来源" name="sources">
                    <pre>{{ JSON.stringify(msg.sources, null, 2) }}</pre>
                  </el-collapse-item>
                </el-collapse>
              </div>
            </div>
          </div>
          <el-empty v-else description="发送消息开始对话" />

          <div class="chat-input-area">
            <el-input
              v-model="message"
              type="textarea"
              :rows="2"
              placeholder="输入消息..."
              :disabled="sending"
              @keydown.ctrl.enter="handleSend"
            />
            <el-button
              type="primary"
              :loading="sending"
              :disabled="!selectedCourseId || !message"
              @click="handleSend"
              style="margin-top: 8px"
            >
              发送
            </el-button>
          </div>
        </el-card>
      </div>

      <!-- Session sidebar -->
      <div class="chat-sidebar">
        <el-card shadow="never">
          <template #header><span>会话列表</span></template>
          <div v-if="sessions.length === 0" class="text-secondary">暂无会话</div>
          <div v-for="s in sessions" :key="s.session_id" class="session-item">
            <div class="session-info">
              <div class="session-intent">
                <el-tag :type="intentTagType(s.intent)" size="small">{{ s.intent }}</el-tag>
              </div>
              <div class="session-status">
                <el-tag
                  :type="s.status === 'completed' ? 'success' : s.status === 'running' ? 'warning' : 'info'"
                  size="small"
                >
                  {{ s.status }}
                </el-tag>
              </div>
            </div>
            <div class="session-meta text-secondary">
              Token: {{ s.token_count }}
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-page {
  height: calc(100vh - var(--header-height) - 48px);
  display: flex;
  flex-direction: column;
}

.chat-layout {
  display: flex;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-messages-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.message-bubble {
  margin-bottom: 16px;
  padding: 12px 16px;
  border-radius: 8px;
  max-width: 85%;
}

.message-bubble.user {
  background: var(--el-color-primary-light-9);
  margin-left: auto;
}

.message-bubble.assistant {
  background: var(--el-fill-color-light);
  margin-right: auto;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.message-content {
  white-space: pre-wrap;
  line-height: 1.6;
  font-size: 14px;
}

.message-sources {
  margin-top: 8px;
}

.message-sources pre {
  font-size: 11px;
  max-height: 200px;
  overflow: auto;
}

.chat-input-area {
  padding-top: 12px;
  border-top: 1px solid var(--el-border-color-light);
}

.chat-sidebar {
  width: 260px;
  flex-shrink: 0;
}

.session-item {
  padding: 8px 0;
  border-bottom: 1px solid var(--el-border-color-light);
}

.session-info {
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 4px;
}

.session-meta {
  font-size: 12px;
}
</style>
