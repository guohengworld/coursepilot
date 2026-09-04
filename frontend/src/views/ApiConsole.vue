<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import api from '@/api/index'
import type { AxiosRequestConfig } from 'axios'

const auth = useAuthStore()

// Request builder
const method = ref<'GET' | 'POST' | 'DELETE'>('GET')
const path = ref('/health')
const bodyText = ref('')
const authHeader = ref(true)
const customHeaders = ref('')
const responseStatus = ref('')
const responseTime = ref('')
const responseSize = ref('')
const responseBody = ref('')
const responseHeaders = ref('')
const sending = ref(false)
const startingTime = ref(0)
const pathParams = ref<Record<string, string>>({})
const showPathParams = ref(false)

// Predefined endpoints
interface Preset {
  label: string
  method: 'GET' | 'POST' | 'DELETE'
  path: string
  body?: string
  hasPathParams?: boolean
}

const presets: Preset[] = [
  { label: '健康检查', method: 'GET', path: '/health' },
  { label: '注册', method: 'POST', path: '/auth/register', body: JSON.stringify({ username: 'test_user', password: 'test_pass_123' }, null, 2) },
  { label: '登录', method: 'POST', path: '/auth/login', body: JSON.stringify({ username: 'test_user', password: 'test_pass_123' }, null, 2) },
  { label: '获取用户', method: 'GET', path: '/auth/me' },
  { label: '课程列表', method: 'GET', path: '/courses' },
  { label: '创建课程', method: 'POST', path: '/courses', body: JSON.stringify({ name: '测试课程', description: '课程描述' }, null, 2) },
  { label: '获取课程', method: 'GET', path: '/courses/{id}', hasPathParams: true },
  { label: '删除课程', method: 'DELETE', path: '/courses/{id}', hasPathParams: true },
  { label: '上传文档', method: 'POST', path: '/courses/upload', body: JSON.stringify({ file: '(选择文件)', course_id: '{id}' }, null, 2) },
  { label: '文档列表', method: 'GET', path: '/courses/{id}/documents', hasPathParams: true },
  { label: '删除文档', method: 'DELETE', path: '/courses/{id}/document/{doc_id}', hasPathParams: true },
  { label: '知识点树', method: 'GET', path: '/courses/{id}/knowledge-points', hasPathParams: true },
  { label: 'RAG 问答', method: 'POST', path: '/courses/{id}/ask', hasPathParams: true, body: JSON.stringify({ question: '什么是进程？' }, null, 2) },
  { label: 'RAG 流式', method: 'POST', path: '/courses/{id}/ask/stream', hasPathParams: true, body: JSON.stringify({ question: '什么是进程？' }, null, 2) },
  { label: 'Agent 对话', method: 'POST', path: '/agent/chat', body: JSON.stringify({ message: '你好', course_id: '{course_id}' }, null, 2) },
  { label: '会话状态', method: 'GET', path: '/agent/sessions/{session_id}', hasPathParams: true },
]

const selectedPreset = ref('')

function onPresetChange(val: string) {
  if (!val) return
  const preset = presets.find((p) => p.label === val)
  if (!preset) return
  method.value = preset.method
  path.value = preset.path
  bodyText.value = preset.body || ''
  responseBody.value = ''
  responseStatus.value = ''
  responseTime.value = ''
  responseSize.value = ''
  responseHeaders.value = ''

  // Extract path params
  const paramMatches = preset.path.match(/\{(\w+)\}/g)
  if (paramMatches) {
    showPathParams.value = true
    const params: Record<string, string> = {}
    paramMatches.forEach((m) => {
      const name = m.replace(/[{}]/g, '')
      params[name] = ''
    })
    pathParams.value = params
  } else {
    showPathParams.value = false
    pathParams.value = {}
  }
}

function resolvePath(): string {
  let p = path.value
  for (const [key, val] of Object.entries(pathParams.value)) {
    if (val) {
      p = p.replace(`{${key}}`, val)
    }
  }
  return p
}

const showBody = computed(() => method.value === 'POST')

async function sendRequest() {
  sending.value = true
  responseBody.value = ''
  responseStatus.value = ''
  responseTime.value = ''
  responseSize.value = ''
  responseHeaders.value = ''
  startingTime.value = Date.now()

  try {
    const headers: Record<string, string> = {}
    if (authHeader.value && auth.token) {
      headers['Authorization'] = `Bearer ${auth.token}`
    }
    // Custom headers
    if (customHeaders.value) {
      customHeaders.value.split('\n').forEach((line) => {
        const idx = line.indexOf(':')
        if (idx > 0) {
          headers[line.slice(0, idx).trim()] = line.slice(idx + 1).trim()
        }
      })
    }

    const resolvedPath = resolvePath()
    const config: AxiosRequestConfig = {
      method: method.value,
      url: resolvedPath,
      headers,
    }

    if (method.value === 'POST') {
      if (resolvedPath === '/courses/upload') {
        // For upload, use FormData
        const formData = new FormData()
        formData.append('course_id', pathParams.value['id'] || '')
        config.data = formData
        delete headers['Content-Type'] // Let browser set boundary
      } else {
        try {
          config.data = JSON.parse(bodyText.value)
          headers['Content-Type'] = 'application/json'
        } catch {
          config.data = bodyText.value
        }
      }
    }

    // For streaming endpoints, show raw event stream
    if (resolvedPath.includes('/ask/stream')) {
      await doStreamRequest(resolvedPath, headers)
      sending.value = false
      return
    }

    const resp = await api.request(config)
    const elapsed = Date.now() - startingTime.value
    responseStatus.value = `${resp.status} ${resp.statusText}`
    responseTime.value = `${elapsed}ms`
    responseBody.value = JSON.stringify(resp.data, null, 2)
    responseSize.value = `${new Blob([responseBody.value]).size} B`
    responseHeaders.value = JSON.stringify(resp.headers as Record<string, unknown>, null, 2)
  } catch (err: unknown) {
    const elapsed = Date.now() - startingTime.value
    responseTime.value = `${elapsed}ms`
    if ((err as any)?.response) {
      const e = err as any
      responseStatus.value = `${e.response.status} ${e.response.statusText}`
      responseBody.value = JSON.stringify(e.response.data, null, 2)
      responseHeaders.value = JSON.stringify(e.response.headers as Record<string, unknown>, null, 2)
    } else if ((err as any)?.request) {
      responseStatus.value = 'No Response'
      responseBody.value = '请求未收到响应（网络错误或超时）'
    } else {
      responseStatus.value = 'Error'
      responseBody.value = (err as Error).message
    }
  }

  sending.value = false
}

async function doStreamRequest(resolvedPath: string, headers: Record<string, string>) {
  responseBody.value = ''
  responseStatus.value = 'Streaming...'

  try {
    const resp = await fetch(`/api/v1${resolvedPath}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(headers.Authorization ? { Authorization: headers.Authorization } : {}),
      },
      body: JSON.stringify(JSON.parse(bodyText.value)),
    })

    responseStatus.value = `${resp.status} ${resp.statusText} (Streaming)`

    const reader = resp.body?.getReader()
    if (!reader) return

    const decoder = new TextDecoder()
    let buffer = ''
    let tokenCount = 0

    // 兼容 JSON 信封（{"t":"tok","c":...} / {"t":"meta"} / {"t":"done"}）
    // 与旧版裸文本 token + "[DONE]"
    const consume = (data: string) => {
      if (data === '[DONE]') {
        responseBody.value += '\n--- [Stream Complete] ---'
        return true
      }
      if (data.startsWith('{')) {
        try {
          const parsed = JSON.parse(data)
          if (parsed.t === 'tok' && typeof parsed.c === 'string') {
            tokenCount++
            responseBody.value += parsed.c
          } else if (parsed.t === 'done') {
            responseBody.value += '\n--- [Stream Complete] ---'
            return true
          }
          // meta 事件不追加正文
          return false
        } catch {
          /* fallthrough：当作裸文本 */
        }
      }
      tokenCount++
      responseBody.value += data
      return false
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (consume(line.slice(line.startsWith('data: ') ? 6 : 0))) {
          return
        }
      }
    }
  } catch (err: unknown) {
    responseBody.value = `Stream error: ${(err as Error).message}`
  }
}

function copyResponse() {
  if (responseBody.value) {
    navigator.clipboard.writeText(responseBody.value)
    ElMessage.success('已复制到剪贴板')
  }
}

const statusColor = computed(() => {
  if (!responseStatus.value) return ''
  if (responseStatus.value.startsWith('2')) return 'success'
  if (responseStatus.value.startsWith('4')) return 'warning'
  if (responseStatus.value.startsWith('5')) return 'danger'
  return 'info'
})
</script>

<template>
  <div class="page-container api-console">
    <div class="page-header">
      <h2>API 控制台</h2>
    </div>

    <!-- Preset selector -->
    <el-card shadow="never" class="mb-16">
      <div class="preset-row">
        <span style="font-weight: 500; white-space: nowrap;">预设端点：</span>
        <el-select
          v-model="selectedPreset"
          placeholder="选择一个端点自动填充..."
          style="flex: 1"
          @change="onPresetChange"
          filterable
        >
          <el-option
            v-for="p in presets"
            :key="p.label"
            :label="`${p.method}  ${p.path}  — ${p.label}`"
            :value="p.label"
          >
            <span>
              <el-tag :type="p.method === 'GET' ? 'success' : p.method === 'POST' ? 'primary' : 'danger'" size="small" style="margin-right: 8px;">
                {{ p.method }}
              </el-tag>
              <code>{{ p.path }}</code>
              <span class="text-secondary" style="margin-left: 8px;">{{ p.label }}</span>
            </span>
          </el-option>
        </el-select>
      </div>
    </el-card>

    <!-- Request Builder -->
    <el-card shadow="never" class="mb-16">
      <div class="request-line">
        <el-select v-model="method" style="width: 120px; margin-right: 8px;">
          <el-option label="GET" value="GET" />
          <el-option label="POST" value="POST" />
          <el-option label="DELETE" value="DELETE" />
        </el-select>
        <el-input
          v-model="path"
          placeholder="/api/v1/{path}"
          class="path-input"
          clearable
        />
        <el-button
          type="primary"
          :loading="sending"
          @click="sendRequest"
          style="margin-left: 8px;"
        >
          {{ sending ? '发送中...' : '发送' }}
        </el-button>
        <el-button @click="() => { responseBody = ''; responseStatus = ''; responseTime = ''; responseSize = ''; responseHeaders = ''; }">
          清空
        </el-button>
      </div>

      <!-- Path params -->
      <el-collapse v-if="showPathParams" style="margin-top: 12px;">
        <el-collapse-item title="路径参数" name="params">
          <div class="params-grid">
            <div v-for="(val, key) in pathParams" :key="key" class="param-item">
              <label>{{ key }}</label>
              <el-input v-model="pathParams[key]" :placeholder="`输入 ${key}`" size="small" />
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>

      <!-- Auth toggle -->
      <div class="options-row" style="margin-top: 12px;">
        <el-checkbox v-model="authHeader" label="添加 Authorization: Bearer 头（已登录）" />
        <el-tag v-if="auth.isLoggedIn" size="small" type="success" effect="plain">已登录</el-tag>
        <el-tag v-else size="small" type="info" effect="plain">未登录</el-tag>
      </div>

      <!-- Custom headers -->
      <div style="margin-top: 8px;">
        <el-input
          v-model="customHeaders"
          type="textarea"
          :rows="2"
          placeholder="自定义请求头（每行一个，格式 Key: Value）&#10;Content-Type: application/json"
        />
      </div>

      <!-- Request body -->
      <div v-if="showBody" style="margin-top: 12px;">
        <label style="font-weight: 500; font-size: 13px; display: block; margin-bottom: 4px;">请求体 (JSON)</label>
        <el-input
          v-model="bodyText"
          type="textarea"
          :rows="8"
          class="code-input"
          placeholder='{"key": "value"}'
        />
      </div>
    </el-card>

    <!-- Response Panel -->
    <el-card shadow="never">
      <template #header>
        <div class="response-header">
          <span>响应</span>
          <div class="response-meta">
            <el-tag v-if="responseStatus" :type="statusColor" size="small" effect="dark">
              {{ responseStatus }}
            </el-tag>
            <span v-if="responseTime" class="text-secondary" style="margin-left: 8px;">{{ responseTime }}</span>
            <span v-if="responseSize" class="text-secondary" style="margin-left: 8px;">{{ responseSize }}</span>
            <el-button v-if="responseBody" size="small" @click="copyResponse" style="margin-left: 8px;">
              复制
            </el-button>
          </div>
        </div>
      </template>

      <div v-if="responseBody" class="response-body">
        <pre><code>{{ responseBody }}</code></pre>
      </div>
      <el-empty v-else description="点击「发送」执行请求" />

      <!-- Response Headers (collapsible) -->
      <el-collapse v-if="responseHeaders">
        <el-collapse-item title="响应头" name="headers">
          <pre class="code-block"><code>{{ responseHeaders }}</code></pre>
        </el-collapse-item>
      </el-collapse>
    </el-card>
  </div>
</template>


<style scoped>
.api-console {
  max-width: 1200px;
}

.preset-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.request-line {
  display: flex;
  align-items: center;
}

.path-input {
  flex: 1;
  font-family: var(--el-font-family-mono, 'Consolas', monospace);
}

.params-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.param-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.param-item label {
  font-weight: 500;
  font-size: 13px;
  min-width: 60px;
  color: var(--el-color-primary);
  font-family: monospace;
}

.options-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.response-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.response-meta {
  display: flex;
  align-items: center;
}

.response-body {
  background: var(--el-fill-color);
  border-radius: 4px;
  padding: 16px;
  overflow: auto;
  max-height: 500px;
}

.response-body pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}

.response-body code {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.5;
  color: var(--el-text-color-primary);
}

.code-input :deep(textarea) {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
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
