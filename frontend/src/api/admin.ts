import { request } from './index'

export interface MemoryLayerStats {
  embedding_coverage: number
  facts_coverage: number
  qa_total: number
  [key: string]: unknown
}

export interface RecallResultItem {
  query: string
  answer: string
  scores: {
    score: number
    recency: number
    relevance: number
    importance: number
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface MemoryDashboardResponse {
  course_id: string
  course_stats: Record<string, unknown>
  memory_layer_stats: MemoryLayerStats
  recent_sessions: Record<string, unknown>[]
}

export interface SessionMemoryResponse {
  session_id: string
  user_id: string
  course_id: string
  intent: string
  status: string
  context_metrics: Record<string, unknown>
  conversation: { role: string; content: string; intent?: string | null }[] | null
  rolling_summary: string | null
  memory_facts: Record<string, unknown> | null
}

export interface MemoryRecallResponse {
  query: string
  results: RecallResultItem[]
}

export function getMemoryDashboard(courseId: string, days: number = 7) {
  return request<MemoryDashboardResponse>('GET', `/admin/memory/dashboard?course_id=${courseId}&days=${days}`)
}

export function getSessionMemory(sessionId: string) {
  return request<SessionMemoryResponse>('GET', `/admin/memory/session/${sessionId}`)
}

export function recallMemory(userId: string, courseId: string, query: string, topK: number = 5) {
  return request<MemoryRecallResponse>(
    'GET',
    `/admin/memory/recall?user_id=${userId}&course_id=${courseId}&query=${encodeURIComponent(query)}&top_k=${topK}`,
  )
}
