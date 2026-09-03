<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const menuItems = [
  { path: '/dashboard', icon: 'Odometer', label: '仪表盘' },
  { path: '/courses', icon: 'Notebook', label: '课程管理' },
  { path: '/knowledge-points', icon: 'Share', label: '知识点树' },
  { path: '/rag-qa', icon: 'ChatLineSquare', label: 'RAG 问答' },
  { path: '/agent', icon: 'MagicStick', label: 'Agent 对话' },
  { path: '/tasks', icon: 'Tickets', label: '任务中心' },
  { path: '/api-console', icon: 'Tools', label: 'API 控制台' },
  ...(auth.isSuperuser ? [{ path: '/admin/memory', icon: 'Cpu', label: '记忆层控制台' }] : []),
]

const activeIndex = computed(() => {
  return menuItems.find((item) => route.path.startsWith(item.path))?.path || '/dashboard'
})

function handleSelect(path: string) {
  router.push(path)
}
</script>

<template>
  <el-menu
    :default-active="activeIndex"
    class="nav-menu"
    @select="handleSelect"
  >
    <div class="nav-brand">
      <el-icon :size="22" color="var(--el-color-primary)"><School /></el-icon>
      <span class="brand-text">CoursePilot</span>
    </div>

    <el-menu-item
      v-for="item in menuItems"
      :key="item.path"
      :index="item.path"
    >
      <el-icon><component :is="item.icon" /></el-icon>
      <span>{{ item.label }}</span>
    </el-menu-item>
  </el-menu>
</template>

<style scoped>
.nav-menu {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  border-right: 1px solid var(--el-border-color-light);
  overflow-y: auto;
  background: var(--el-bg-color);
  z-index: 100;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 18px 20px;
  border-bottom: 1px solid var(--el-border-color-light);
  margin-bottom: 8px;
}

.brand-text {
  font-size: 18px;
  font-weight: 700;
  color: var(--el-text-color-primary);
  letter-spacing: 0.5px;
}
</style>
