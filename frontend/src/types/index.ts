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
  document_id: string | null
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
  session_id?: string
}

export interface ChatAcceptedResponse {
  session_id: string
  status: string
}

export interface DiagnosisData {
  overall_rate: number
  total_practiced: number
  kp_stats: Record<string, { total: number; correct: number; rate: number }>
  weak_kps: string[]
  llm_analysis: string
  recommendations: string
}

export interface SessionPollResponse {
  session_id: string
  course_id: string
  status: string
  intent: string
  query: string | null
  answer: string | null
  sources: Record<string, unknown>[] | null
  questions: QuizQuestion[] | null
  diagnosis_data: DiagnosisData | null
  token_count: number
  estimated_cost: number
  conversation: { role: string; content: string; intent?: string | null }[] | null
  created_at: string
  updated_at: string
}

export interface SessionListItem {
  session_id: string
  course_id: string
  intent: string
  status: string
  query: string | null
  token_count: number
  estimated_cost: number
  created_at: string
  updated_at: string
}

// ===== Common =====
export interface ApiError {
  detail: string
}

// ===== Practice =====
export interface QuizQuestion {
  index: number
  question_text: string
  options: Record<string, string>
  kp_path: string
}

export interface QuizResponse {
  session_id: string
  questions: QuizQuestion[]
}

export interface SubmitRequest {
  answers: Record<string, string>
}

export interface QuestionResult {
  index: number
  question_text: string
  correct: boolean
  student_answer: string
  correct_answer: string
  explanation: string
  kp_path: string
}

export interface SubmitResponse {
  session_id: string
  total: number
  correct: number
  score: number
  results: QuestionResult[]
}

export interface ApiResult<T> {
  ok: boolean
  status: number
  data: T | ApiError
}
