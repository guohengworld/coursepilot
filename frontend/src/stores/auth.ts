import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { UserInfo } from '@/types'
import { getMe } from '@/api/auth'

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('token'))
  const tokenExpiresAt = ref<number | null>(Number(localStorage.getItem('token_expires_at')) || null)
  const user = ref<UserInfo | null>(null)
  const loaded = ref(false)

  const isLoggedIn = computed(() => {
    if (!token.value) return false
    if (tokenExpiresAt.value && Date.now() > tokenExpiresAt.value) {
      logout()
      return false
    }
    return true
  })

  const isSuperuser = computed(() => user.value?.role === 'super')
  const isTeacher = computed(() => user.value?.role === 'teacher' || user.value?.role === 'super')

  function setToken(newToken: string, expiresIn: number) {
    token.value = newToken
    tokenExpiresAt.value = Date.now() + expiresIn * 1000
    localStorage.setItem('token', newToken)
    localStorage.setItem('token_expires_at', String(tokenExpiresAt.value))
  }

  function setUser(u: UserInfo) {
    user.value = u
  }

  function logout() {
    token.value = null
    tokenExpiresAt.value = null
    user.value = null
    loaded.value = false
    localStorage.removeItem('token')
    localStorage.removeItem('token_expires_at')
  }

  async function fetchUser() {
    if (!token.value) return false
    const res = await getMe()
    if (res.ok && !('detail' in res.data)) {
      user.value = res.data as UserInfo
      loaded.value = true
      return true
    }
    logout()
    return false
  }

  return {
    token,
    tokenExpiresAt,
    user,
    loaded,
    isLoggedIn,
    isSuperuser,
    isTeacher,
    setToken,
    setUser,
    logout,
    fetchUser,
  }
})
