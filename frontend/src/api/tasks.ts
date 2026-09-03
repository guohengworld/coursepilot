import { request } from './index'
import type {
  TaskCandidate,
  TaskDetail,
  TaskDraftUpdate,
  TaskListItem,
} from '@/types'

function buildQuery(params?: Record<string, string | undefined>): string {
  const qs = new URLSearchParams()
  for (const [key, val] of Object.entries(params ?? {})) {
    if (val) qs.set(key, val)
  }
  const s = qs.toString()
  return s ? `?${s}` : ''
}

/** 任务列表（后端按角色分流：teacher=自己创建的 / student=发布给自己的 / super=全部） */
export function getTasks(params?: { course_id?: string; status?: string }) {
  return request<TaskListItem[]>(
    'GET',
    `/tasks${buildQuery({ course_id: params?.course_id, status: params?.status })}`,
  )
}

/** 教师选人名单：该课程 enrollments role=student 的学生 */
export function getTaskCandidates(courseId: string) {
  return request<TaskCandidate[]>('GET', `/tasks/candidates?course_id=${courseId}`)
}

/** 生成任务草稿（同步等待 LLM 生成后落库，返回完整草稿） */
export function createTaskDraft(data: { course_id: string; student_id: string }) {
  return request<TaskDetail>('POST', '/tasks/draft', data)
}

/** 任务详情 */
export function getTask(id: string) {
  return request<TaskDetail>('GET', `/tasks/${id}`)
}

/** 字段级编辑草稿（goal/groups/time_limit_minutes/acceptance，diagnosis 不可改） */
export function updateTask(id: string, data: TaskDraftUpdate) {
  return request<TaskDetail>('PUT', `/tasks/${id}`, data)
}

/** 发布草稿 → 学生端立即可见 */
export function publishTask(id: string) {
  return request<TaskDetail>('POST', `/tasks/${id}/publish`)
}
