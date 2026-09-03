import { createRouter, createWebHashHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/',
    redirect: '/dashboard',
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/courses',
    name: 'Courses',
    component: () => import('@/views/Courses.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/courses/:id',
    name: 'CourseDetail',
    component: () => import('@/views/CourseDetail.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/knowledge-points',
    name: 'KnowledgePoints',
    component: () => import('@/views/KnowledgePoints.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/rag-qa',
    name: 'RagQA',
    component: () => import('@/views/RagQA.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/agent',
    name: 'AgentChat',
    component: () => import('@/views/AgentChat.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: () => import('@/views/Tasks.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/api-console',
    name: 'ApiConsole',
    component: () => import('@/views/ApiConsole.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/admin/memory',
    name: 'AdminMemoryConsole',
    component: () => import('@/views/AdminMemoryConsole.vue'),
    meta: { requiresAuth: true, requiresAdmin: true },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

router.beforeEach((to, _from, next) => {
  const auth = useAuthStore()

  if (to.meta.requiresAuth === false) {
    next()
    return
  }

  if (!auth.isLoggedIn) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
    return
  }

  if (to.meta.requiresAdmin && !auth.isSuperuser) {
    next({ name: 'Dashboard' })
    return
  }

  next()
})

export default router
