// ===== Auth =====
export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
}

export interface LoginResponse {
  token: string
  expires_in: number
  user: UserInfo
}

export interface RegisterResponse {
  user_id: string
  username: string
  role: string
}

export interface UserInfo {
  id: string
  username: string
  role: string
  created_at: string
}

// ===== Course =====
export interface CourseCreate {
  name: string
  description?: string
}

export interface Course {
  id: string
  name: string
  description: string | null
  created_by: string
  created_at: string
}

export interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number | null
  status: string
  page_count: number | null
  uploaded_at: string
}

export interface KnowledgePoint {
  id: string
  parent_id: string | null
  kp_path: string
  title: string
  summary: string | null
  difficulty: number | null
  sort_order: number | null
}

// ===== RAG =====
export interface AskRequest {
  question: string
}

export interface AskResponse {
  answer: string
  trace_id: string
  rewritten_query: string
  citations: number[]
  top_scores: number[]
  source_kp_paths: string[]
}

// ===== Agent =====
export interface ChatRequest {
  message: string
  course_id: string
}

export interface ChatResponse {
  session_id: string
  intent: string
  answer: string
  sources: Record<string, unknown>[]
  token_count: number
}

export interface SessionStatus {
  session_id: string
  intent: string
  status: string
  token_count: number
  estimated_cost: number
  created_at: string
  updated_at: string
}

// ===== Common =====
export interface ApiError {
  detail: string
}

export interface ApiResult<T> {
  ok: boolean
  status: number
  data: T | ApiError
}
