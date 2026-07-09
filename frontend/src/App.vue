<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import NavBar from '@/components/NavBar.vue'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()
const initialized = ref(false)

onMounted(async () => {
  if (auth.isLoggedIn) {
    await auth.fetchUser()
  }
  initialized.value = true
})

watch(
  () => auth.isLoggedIn,
  (loggedIn) => {
    if (loggedIn && !auth.user) {
      auth.fetchUser()
    }
  },
)
</script>

<template>
  <div class="app-layout" v-if="initialized">
    <template v-if="auth.isLoggedIn">
      <NavBar />
      <div class="main-area">
        <header class="top-bar">
          <div class="top-bar-left">
            <span class="page-title">{{ route.meta?.title || '' }}</span>
          </div>
          <div class="top-bar-right">
            <el-dropdown trigger="click" v-if="auth.user">
              <span class="user-info">
                <el-icon :size="18"><UserFilled /></el-icon>
                {{ auth.user.username }}
                <el-tag
                  :type="auth.user.role === 'super' ? 'danger' : 'info'"
                  size="small"
                  effect="plain"
                  style="margin-left:4px"
                >
                  {{ auth.user.role }}
                </el-tag>
                <el-icon class="el-icon--right"><ArrowDown /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="auth.logout(); router.push('/login')">
                    <el-icon><SwitchButton /></el-icon> 退出登录
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </header>
        <main class="main-content">
          <router-view />
        </main>
      </div>
    </template>
    <template v-else>
      <div class="fullscreen-center">
        <router-view />
      </div>
    </template>
  </div>
  <div class="app-loading" v-else>
    <el-icon class="is-loading" :size="32"><Loading /></el-icon>
  </div>
</template>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  background: var(--el-bg-color-page);
}

.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  margin-left: var(--sidebar-width);
  overflow: hidden;
}

.top-bar {
  height: var(--header-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-light);
  flex-shrink: 0;
}

.page-title {
  font-size: 16px;
  font-weight: 500;
  color: var(--el-text-color-primary);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 14px;
  color: var(--el-text-color-regular);
}

.main-content {
  flex: 1;
  overflow: auto;
  padding: 24px;
}

.fullscreen-center {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--el-bg-color-page);
}

.app-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
}
</style>
