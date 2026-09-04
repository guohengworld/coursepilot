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
  /** 当前用户在本课程内的角色：'teacher' | 'student' | null（未加入） */
  role?: string | null
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

/** 引用 id 对应的教材来源（与后端 citation_map 对齐） */
export interface CitationRef {
  uuid?: string
  kp_path?: string
  page_ref?: string
}

export interface AskResponse {
  answer: string
  trace_id: string
  rewritten_query: string
  citations: number[]
  top_scores: number[]
  source_kp_paths: string[]
  /** ref id → {uuid, kp_path, page_ref}，供前端渲染引用来源面板 */
  citation_map?: Record<string, CitationRef>
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

/** Agent 会话中的一轮对话 */
export interface ConversationTurn {
  role: string
  content: string
  intent?: string | null
  sources?: Record<string, unknown>[] | null
  /** 该轮回答的 ref id → 来源映射（后端在 assistant 轮次中持久化） */
  citations?: Record<string, CitationRef> | null
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
  conversation: ConversationTurn[] | null
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

// ===== Tasks（⑤ 教师发布任务，schema 与后端 api/tasks.py 对齐） =====
export interface TaskGoal {
  metric?: string
  description?: string
  [key: string]: unknown
}

export interface TaskGroup {
  kp_path: string
  kp_name: string
  question_count: number
  difficulty: number
  source: string | null
  reason: string | null
}

export interface TaskAcceptance {
  pass_condition?: string
  fallback_action?: string | null
  [key: string]: unknown
}

export interface TaskDiagnosis {
  mastery_level?: Record<string, unknown>
  weak_kps?: string[]
  common_mistakes?: unknown[]
  avg_correct_rate?: number | null
  class_rank?: string | null
  [key: string]: unknown
}

export type TaskStatus = 'draft' | 'published'

export interface TaskListItem {
  id: string
  course_id: string
  student_id: string
  status: TaskStatus
  goal: TaskGoal
  total_count: number
  time_limit_minutes: number | null
  created_at: string
  updated_at: string
  published_at: string | null
}

export interface TaskDetail extends TaskListItem {
  created_by: string
  diagnosis: TaskDiagnosis
  groups: TaskGroup[]
  acceptance: TaskAcceptance
}

export interface TaskCandidate {
  user_id: string
  username: string
}

export interface TaskDraftUpdate {
  goal?: TaskGoal | null
  groups?: TaskGroup[] | null
  time_limit_minutes?: number | null
  acceptance?: TaskAcceptance | null
}
