<script setup lang="ts">
import { ref, onMounted, onUnmounted, reactive } from 'vue'
import { useRoute } from 'vue-router'
import { getCourses, chat, getSession, listSessions, approveSession, deleteSession } from '@/api'
import { getQuiz, submitAnswers } from '@/api/practice'
import type { Course, SessionListItem } from '@/types'
import type { ChatAcceptedResponse, SessionPollResponse, QuizQuestion, SubmitResponse, DiagnosisData } from '@/types'

const route = useRoute()
import { useAuthStore } from '@/stores/auth'
const authStore = useAuthStore()
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
  quiz?: QuizQuestion[]
  answers: Record<string, string>
  submitted: boolean
  submitResult?: SubmitResponse
  diagnosis?: DiagnosisData
}

// 多会话隔离状态
const messages = ref<Message[]>([])
const activeSessionId = ref<string | null>(null)
const sessionsList = ref<SessionListItem[]>([])
const submittingMap = reactive<Record<string, boolean>>({})
let pollTimer: ReturnType<typeof setInterval> | null = null

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

async function loadCourses() {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
}

async function loadSessions() {
  const res = await listSessions()
  if (res.ok && Array.isArray(res.data)) {
    sessionsList.value = res.data as SessionListItem[]
  }
}

function handleNewSession() {
  if (pollTimer) clearInterval(pollTimer)
  activeSessionId.value = null
  messages.value = []
}

async function selectSession(sessionId: string) {
  if (sending.value) return  // 处理中不允许切换
  if (pollTimer) clearInterval(pollTimer)

  const existing = sessionsList.value.find(s => s.session_id === sessionId)
  if (!existing) return
  if (activeSessionId.value === sessionId) return

  // 回填课程选择
  selectedCourseId.value = existing.course_id

  // 从 API 获取会话详情
  const res = await getSession(sessionId)
  if (!res.ok) return

  const data = res.data as SessionPollResponse

  // 从 conversation 构建消息列表（多轮对话）
  const msgs: Message[] = []
  if (data.conversation && data.conversation.length > 0) {
    let currentQuiz: QuizQuestion[] | undefined
    for (const turn of data.conversation) {
      if (turn.role === 'user') {
        msgs.push({ role: 'user', content: turn.content, answers: {}, submitted: false })
      } else if (turn.role === 'assistant') {
        const msg: Message = {
          role: 'assistant',
          content: turn.content,
          answers: {},
          submitted: false,
          intent: turn.intent || undefined,
          sessionId: sessionId,
        }
        // 只有最后一条 assistant 消息可能附带题目或诊断
        if (turn.intent === 'practice') {
          currentQuiz = data.questions || undefined
        }
        msgs.push(msg)
      }
    }
    // 将题目挂到最后一条 assistant 消息
    if (currentQuiz) {
      const lastAssistant = msgs.filter(m => m.role === 'assistant').pop()
      if (lastAssistant) {
        lastAssistant.quiz = currentQuiz
        currentQuiz.forEach((_, qi) => { lastAssistant.answers[String(qi)] = '' })
      }
    }
    // 将诊断数据挂到最后一条 assistant 消息
    if (data.intent === 'diagnose' && data.diagnosis_data) {
      const lastAssistant = msgs.filter(m => m.role === 'assistant').pop()
      if (lastAssistant) {
        lastAssistant.diagnosis = data.diagnosis_data
      }
    }
  } else {
    // 降级：无 conversation 时用 query + answer
    if (data.query) {
      msgs.push({ role: 'user', content: data.query, answers: {}, submitted: false })
    }
    const assistantMsg: Message = {
      role: 'assistant',
      content: data.answer || '',
      answers: {},
      submitted: false,
      intent: data.intent || undefined,
      sessionId: data.session_id,
    }
    if (data.sources) assistantMsg.sources = data.sources
    if (data.intent === 'practice' && data.questions) {
      assistantMsg.quiz = data.questions
      data.questions.forEach((_, qi) => { assistantMsg.answers[String(qi)] = '' })
    }
    if (data.intent === 'diagnose' && data.diagnosis_data) {
      assistantMsg.diagnosis = data.diagnosis_data
    }
    msgs.push(assistantMsg)
  }

  messages.value = msgs
  activeSessionId.value = sessionId
}

async function handleSend() {
  if (!selectedCourseId.value || !message.value) return

  if (pollTimer) clearInterval(pollTimer)

  const userMsg = message.value
  message.value = ''
  sending.value = true

  // 追加用户消息到当前对话
  const msgs = [...messages.value,
    { role: 'user' as const, content: userMsg, answers: {}, submitted: false },
  ]
  messages.value = msgs

  // 提交请求（带 session_id 继续已有会话）
  const res = await chat({
    message: userMsg,
    course_id: selectedCourseId.value,
    session_id: activeSessionId.value || undefined,
  })
  if (!res.ok) {
    msgs.push({
      role: 'assistant',
      content: `错误: ${(res.data as any)?.detail || '请求失败'}`,
      answers: {},
      submitted: false,
    })
    sending.value = false
    return
  }

  const accepted = res.data as ChatAcceptedResponse
  const sessionId = accepted.session_id
  activeSessionId.value = sessionId

  // 创建 placeholder 用于轮询
  const placeholder: Message = {
    role: 'assistant',
    content: '正在处理...',
    answers: {},
    submitted: false,
  }
  msgs.push(placeholder)

  // 轮询结果
  let pollCount = 0
  const MAX_POLL = 120
  pollTimer = setInterval(async () => {
    pollCount++
    if (pollCount > MAX_POLL) {
      clearInterval(pollTimer!)
      pollTimer = null
      placeholder.content = '错误: 处理超时，请重试'
      sending.value = false
      return
    }

    const sr = await getSession(sessionId)
    if (!sr.ok) {
      clearInterval(pollTimer!)
      pollTimer = null
      placeholder.content = `错误: ${(sr.data as any)?.detail || '查询失败'}`
      sending.value = false
      return
    }

    const data = sr.data as SessionPollResponse

    if (data.status === 'processing') {
      placeholder.content = '正在处理' + '.'.repeat(pollCount % 4)
      return
    }

    if (data.status === 'waiting_human') {
      placeholder.content = '⏳ 等待教师审批中...\n会话已暂停，待教师确认后将自动继续'
      return
    }

    // 终态：完成 / 拒绝 / 失败
    clearInterval(pollTimer!)
    pollTimer = null
    sending.value = false

    if (data.status === 'rejected') {
      placeholder.content = data.answer || '操作已被管理员拒绝'
      placeholder.intent = data.intent
      placeholder.sessionId = data.session_id
      await loadSessions()
      return
    }

    if (data.status === 'failed') {
      placeholder.content = '错误: 处理失败，请重试'
      return
    }

    // status === 'completed'
    placeholder.content = data.answer || ''
    placeholder.intent = data.intent
    placeholder.sources = data.sources || undefined
    placeholder.sessionId = data.session_id

    // 练习：从轮询结果加载题目
    if (data.intent === 'practice' && data.questions) {
      placeholder.quiz = data.questions
      data.questions.forEach((_, qi) => {
        placeholder.answers[String(qi)] = ''
      })
    }

    // 诊断：加载结构化数据
    if (data.intent === 'diagnose' && data.diagnosis_data) {
      placeholder.diagnosis = data.diagnosis_data
    }

    // 刷新会话列表
    await loadSessions()
  }, 1000)
}

async function loadQuiz(sessionId: string, msg: Message) {
  const res = await getQuiz(sessionId)
  if (res.ok && !('detail' in res.data)) {
    const data = res.data as { session_id: string; questions: QuizQuestion[] }
    msg.quiz = data.questions
  }
}

async function handleSubmitQuiz(msg: Message) {
  if (!msg.sessionId || msg.submitted) return
  submittingMap[msg.sessionId] = true

  const res = await submitAnswers(msg.sessionId, { answers: msg.answers })
  if (res.ok && !('detail' in res.data)) {
    msg.submitResult = res.data as SubmitResponse
    msg.submitted = true
  } else {
    msg.content += `\n\n提交失败: ${(res.data as any)?.detail || '请求失败'}`
  }
  submittingMap[msg.sessionId] = false
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

function resultIcon(correct: boolean) {
  return correct ? '✅' : '❌'
}

async function handleDeleteSession(sessionId: string) {
  const res = await deleteSession(sessionId)
  if (res.ok) {
    // 从列表中移除
    sessionsList.value = sessionsList.value.filter(s => s.session_id !== sessionId)
    // 如果删除的是当前会话，清空
    if (activeSessionId.value === sessionId) {
      handleNewSession()
    }
  }
}

async function handleApprove(sessionId: string, approved: boolean) {
  const res = await approveSession(sessionId, approved)
  if (res.ok) {
    await loadSessions()
    // 如果是当前激活的会话且被拒绝，更新显示
    if (activeSessionId.value === sessionId) {
      const sr = await getSession(sessionId)
      if (sr.ok) {
        const data = sr.data as SessionPollResponse
        if (data.status === 'rejected') {
          const lastMsg = messages.value[messages.value.length - 1]
          if (lastMsg && lastMsg.role === 'assistant') {
            lastMsg.content = data.answer || '操作已被管理员拒绝'
          }
        }
      }
    }
  }
}

onMounted(() => {
  loadCourses()
  loadSessions()
})
</script>

<template>
  <div class="page-container chat-page">
    <div class="page-header">
      <h2>Agent 对话</h2>
      <el-button type="primary" @click="handleNewSession" :disabled="sending">
        新建会话
      </el-button>
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

              <!-- 消息内容 -->
              <div class="message-content">{{ msg.content }}</div>

              <!-- 练习交互区域（仅 practice intent） -->
              <div v-if="msg.quiz && msg.intent === 'practice'" class="quiz-area">
                <el-divider />
                <div v-for="(q, qi) in msg.quiz" :key="qi" class="quiz-question">
                  <div class="quiz-question-text">{{ qi + 1 }}. {{ q.question_text }}</div>

                  <!-- 提交前：radio 选择 -->
                  <template v-if="!msg.submitted">
                    <el-radio-group
                      v-model="msg.answers[String(qi)]"
                      class="quiz-options"
                    >
                      <el-radio
                        v-for="(optText, optKey) in q.options"
                        :key="optKey"
                        :label="optKey"
                        class="quiz-option"
                      >
                        {{ optKey }}. {{ optText }}
                      </el-radio>
                    </el-radio-group>
                  </template>

                  <!-- 提交后：显示结果 -->
                  <template v-else>
                    <div class="quiz-result">
                      <div
                        v-for="(optText, optKey) in q.options"
                        :key="optKey"
                        :class="[
                          'quiz-option-result',
                          {
                            'correct-option': msg.submitResult?.results[qi]?.correct_answer === optKey,
                            'wrong-option': msg.answers[String(qi)] === optKey && msg.submitResult?.results[qi]?.correct_answer !== optKey,
                          }
                        ]"
                      >
                        <span class="result-icon">
                          {{ msg.submitResult?.results[qi]?.correct_answer === optKey ? '✅' : '' }}
                          {{ msg.answers[String(qi)] === optKey && msg.submitResult?.results[qi]?.correct_answer !== optKey ? '❌' : '' }}
                        </span>
                        {{ optKey }}. {{ optText }}
                      </div>
                    </div>
                    <div
                      :class="['quiz-feedback', msg.submitResult?.results[qi]?.correct ? 'correct' : 'wrong']"
                    >
                      <template v-if="msg.submitResult?.results[qi]?.correct">
                        ✅ 正确
                      </template>
                      <template v-else>
                        ❌ 错误，正确答案：{{ msg.submitResult?.results[qi]?.correct_answer }}
                      </template>
                      — {{ msg.submitResult?.results[qi]?.explanation }}
                    </div>
                  </template>
                </div>

                <!-- 提交按钮 -->
                <div v-if="!msg.submitted" class="quiz-actions">
                  <el-button
                    type="primary"
                    :loading="submittingMap[msg.sessionId!]"
                    :disabled="!Object.keys(msg.answers).length"
                    @click="handleSubmitQuiz(msg)"
                  >
                    提交答案
                  </el-button>
                </div>

                <!-- 提交后的分数 -->
                <div v-if="msg.submitResult" class="quiz-score">
                  <el-tag
                    :type="msg.submitResult.score >= 0.6 ? 'success' : 'warning'"
                    size="large"
                  >
                    得分：{{ msg.submitResult.correct }}/{{ msg.submitResult.total }}
                    ({{ (msg.submitResult.score * 100).toFixed(0) }}%)
                  </el-tag>
                </div>
              </div>

              <!-- 学情诊断报告 -->
              <div v-if="msg.diagnosis" class="diagnosis-area">
                <el-divider />
                <!-- 整体概览 -->
                <div class="diag-summary">
                  <el-tag
                    :type="msg.diagnosis.overall_rate >= 0.6 ? 'success' : 'warning'"
                    size="large"
                  >
                    正确率：{{ (msg.diagnosis.overall_rate * 100).toFixed(0) }}%
                    （{{ msg.diagnosis.total_practiced }} 题）
                  </el-tag>
                </div>

                <!-- 各知识点详情 -->
                <div v-if="msg.diagnosis.kp_stats" class="diag-kp-stats">
                  <div class="diag-section-title">📊 知识点掌握详情</div>
                  <div
                    v-for="(stat, kpPath) in msg.diagnosis.kp_stats"
                    :key="kpPath"
                    class="diag-kp-row"
                  >
                    <div class="diag-kp-path" :title="kpPath">
                      {{ kpPath.split('/').pop() }}
                    </div>
                    <div class="diag-kp-bar-wrapper">
                      <el-progress
                        :percentage="Math.round(stat.rate * 100)"
                        :status="stat.rate >= 0.6 ? 'success' : 'exception'"
                        :stroke-width="18"
                        :text-inside="true"
                      />
                    </div>
                    <div class="diag-kp-num">{{ stat.correct }}/{{ stat.total }}</div>
                  </div>
                </div>

                <!-- 薄弱知识点 -->
                <div v-if="msg.diagnosis.weak_kps && msg.diagnosis.weak_kps.length" class="diag-weak-section">
                  <div class="diag-section-title">⚠️ 薄弱知识点</div>
                  <div class="diag-weak-tags">
                    <el-tag
                      v-for="kp in msg.diagnosis.weak_kps"
                      :key="kp"
                      type="danger"
                      size="small"
                    >
                      {{ kp.split('/').pop() }}
                    </el-tag>
                  </div>
                </div>

                <!-- LLM 深度分析 -->
                <div v-if="msg.diagnosis.llm_analysis" class="diag-analysis">
                  <div class="diag-section-title">📝 深度分析</div>
                  <div class="diag-analysis-text">{{ msg.diagnosis.llm_analysis }}</div>
                </div>

                <!-- 学习建议 -->
                <div v-if="msg.diagnosis.recommendations" class="diag-recommendations">
                  <div class="diag-section-title">📌 学习建议</div>
                  <div class="diag-recommendations-text">{{ msg.diagnosis.recommendations }}</div>
                </div>
              </div>

              <!-- 来源折叠面板 -->
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
          <template #header>
            <span>历史会话</span>
            <el-badge
              v-if="authStore.isTeacher"
              :value="sessionsList.filter(s => s.status === 'waiting_human').length"
              :hidden="sessionsList.filter(s => s.status === 'waiting_human').length === 0"
              class="pending-badge"
            />
          </template>
          <div v-if="sessionsList.length === 0" class="text-secondary">暂无会话</div>
          <div
            v-for="s in sessionsList"
            :key="s.session_id"
            :class="['session-item', { 'session-active': s.session_id === activeSessionId }]"
            @click="selectSession(s.session_id)"
          >
            <div class="session-info">
              <div class="session-intent">
                <el-tag :type="intentTagType(s.intent)" size="small">{{ s.intent }}</el-tag>
              </div>
              <div class="session-status">
                <el-tag
                  :type="s.status === 'completed' ? 'success' : (s.status === 'processing' || s.status === 'running') ? 'warning' : s.status === 'waiting_human' ? 'info' : s.status === 'rejected' ? 'danger' : 'info'"
                  size="small"
                >
                  {{ s.status === 'waiting_human' ? '等待审批' : s.status === 'rejected' ? '已拒绝' : s.status }}
                </el-tag>
              </div>
              <el-button
                size="small"
                text
                type="danger"
                class="session-delete-btn"
                @click.stop="handleDeleteSession(s.session_id)"
                title="删除会话"
              >
                ✕
              </el-button>
            </div>
            <div class="session-query text-secondary" :title="s.query || ''">
              {{ s.query || '（无提问）' }}
            </div>
            <!-- 教师审批按钮 -->
            <div
              v-if="s.status === 'waiting_human' && authStore.isTeacher"
              class="session-approve-actions"
              @click.stop
            >
              <el-button size="small" type="success" @click="handleApprove(s.session_id, true)">
                批准
              </el-button>
              <el-button size="small" type="danger" @click="handleApprove(s.session_id, false)">
                拒绝
              </el-button>
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
  width: 280px;
  flex-shrink: 0;
  overflow-y: auto;
}

.session-item {
  padding: 10px 8px;
  border-bottom: 1px solid var(--el-border-color-light);
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.2s;
  margin-bottom: 2px;
}

.session-item:hover {
  background: var(--el-color-primary-light-9);
}

.session-active {
  background: var(--el-color-primary-light-8);
}

.session-info {
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 4px;
}

.session-query {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.session-approve-actions {
  display: flex;
  gap: 4px;
  margin-top: 6px;
}

.pending-badge {
  margin-left: 4px;
}

/* Quiz 交互区域 */
.quiz-area {
  margin-top: 8px;
}

.quiz-question {
  margin-bottom: 16px;
}

.quiz-question-text {
  font-weight: 600;
  margin-bottom: 8px;
  font-size: 14px;
}

.quiz-options {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.quiz-option {
  margin: 2px 0;
}

.quiz-result {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.quiz-option-result {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 13px;
}

.correct-option {
  background: var(--el-color-success-light-9);
  color: var(--el-color-success);
}

.wrong-option {
  background: var(--el-color-danger-light-9);
  color: var(--el-color-danger);
}

.result-icon {
  margin-right: 4px;
}

.quiz-feedback {
  margin-top: 4px;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 13px;
  line-height: 1.5;
}

.quiz-feedback.correct {
  background: var(--el-color-success-light-9);
  color: var(--el-color-success);
}

.quiz-feedback.wrong {
  background: var(--el-color-danger-light-9);
  color: var(--el-color-danger);
}

.quiz-actions {
  margin-top: 12px;
}

.quiz-score {
  margin-top: 12px;
  text-align: center;
}

/* 学情诊断区域 */
.diagnosis-area {
  margin-top: 8px;
}

.diag-summary {
  margin-bottom: 12px;
  text-align: center;
}

.diag-section-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
  color: var(--el-text-color-primary);
}

.diag-kp-stats {
  margin-bottom: 12px;
}

.diag-kp-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.diag-kp-path {
  font-size: 12px;
  min-width: 80px;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--el-text-color-secondary);
}

.diag-kp-bar-wrapper {
  flex: 1;
}

.diag-kp-num {
  font-size: 12px;
  min-width: 40px;
  text-align: right;
  color: var(--el-text-color-secondary);
}

.diag-weak-section {
  margin-bottom: 12px;
}

.diag-weak-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.diag-analysis,
.diag-recommendations {
  margin-bottom: 12px;
}

.diag-analysis-text,
.diag-recommendations-text {
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  color: var(--el-text-color-regular);
  padding: 8px 12px;
  background: var(--el-fill-color-lighter);
  border-radius: 6px;
}
</style>
