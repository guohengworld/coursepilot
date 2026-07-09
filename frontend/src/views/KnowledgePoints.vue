<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { getCourses, getDocuments, getKnowledgePoints } from '@/api/courses'
import type { Course, Document, KnowledgePoint } from '@/types'

const route = useRoute()

const courses = ref<Course[]>([])
const documents = ref<Document[]>([])
const selectedCourseId = ref(route.query.courseId as string || '')
const selectedDocumentId = ref('')
const flatNodes = ref<KnowledgePoint[]>([])
const loading = ref(false)

interface TreeNode extends KnowledgePoint {
  children: TreeNode[]
}

const treeData = computed(() => {
  return buildTree(flatNodes.value)
})

function buildTree(nodes: KnowledgePoint[]): TreeNode[] {
  const map = new Map<string, TreeNode>()
  const roots: TreeNode[] = []

  nodes.forEach((n) => {
    map.set(n.id, { ...n, children: [] })
  })

  nodes.forEach((n) => {
    const node = map.get(n.id)!
    if (n.parent_id && map.has(n.parent_id)) {
      map.get(n.parent_id)!.children.push(node)
    } else {
      roots.push(node)
    }
  })

  return roots
}

async function loadCourses() {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
}

async function loadDocuments() {
  if (!selectedCourseId.value) {
    documents.value = []
    return
  }
  const res = await getDocuments(selectedCourseId.value)
  if (res.ok && Array.isArray(res.data)) {
    documents.value = res.data as Document[]
  }
}

async function loadKnowledgePoints() {
  if (!selectedCourseId.value) return
  loading.value = true
  const res = await getKnowledgePoints(
    selectedCourseId.value,
    selectedDocumentId.value || undefined,
  )
  if (res.ok && Array.isArray(res.data)) {
    flatNodes.value = res.data as KnowledgePoint[]
  }
  loading.value = false
}

function handleCourseChange(val: string) {
  selectedCourseId.value = val
  selectedDocumentId.value = ''
  documents.value = []
  flatNodes.value = []
  if (val) {
    loadDocuments()
  }
  loadKnowledgePoints()
}

function handleDocumentChange() {
  flatNodes.value = []
  loadKnowledgePoints()
}

onMounted(() => {
  loadCourses()
  if (selectedCourseId.value) {
    loadDocuments()
    loadKnowledgePoints()
  }
})
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2>知识点树</h2>
    </div>

    <div class="filters mb-16">
      <el-select
        v-model="selectedCourseId"
        placeholder="请选择课程"
        style="width: 280px"
        @change="handleCourseChange"
      >
        <el-option
          v-for="c in courses"
          :key="c.id"
          :label="c.name"
          :value="c.id"
        />
      </el-select>

      <el-select
        v-model="selectedDocumentId"
        placeholder="全部教材"
        style="width: 280px"
        clearable
        @change="handleDocumentChange"
      >
        <el-option
          v-for="d in documents"
          :key="d.id"
          :label="d.filename"
          :value="d.id"
        />
      </el-select>
    </div>

    <el-card v-loading="loading" shadow="never">
      <template v-if="treeData.length > 0">
        <el-tree
          :data="treeData"
          node-key="id"
          :props="{
            children: 'children',
            label: 'title',
          }"
          default-expand-all
          :filter-node-method="(value: string, data: any) => data.title.includes(value)"
        >
          <template #default="{ node, data }">
            <span class="tree-node">
              <span class="tree-label">{{ data.title }}</span>
              <span v-if="data.difficulty !== null" class="tree-difficulty">
                <el-tag size="small" :type="data.difficulty > 3 ? 'danger' : data.difficulty > 2 ? 'warning' : 'success'">
                  难度 {{ data.difficulty }}
                </el-tag>
              </span>
              <span class="tree-path text-secondary">{{ data.kp_path }}</span>
            </span>
          </template>
        </el-tree>
      </template>
      <el-empty v-else-if="!loading && selectedCourseId" description="暂无知识点数据" />
      <el-empty v-else-if="!loading" description="请先选择课程" />
    </el-card>
  </div>
</template>

<style scoped>
.filters {
  display: flex;
  gap: 12px;
  align-items: center;
}

.tree-node {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.tree-label {
  font-weight: 500;
}

.tree-path {
  font-size: 12px;
  margin-left: 4px;
}

.tree-difficulty {
  flex-shrink: 0;
}
</style>

<style scoped>
.tree-node {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.tree-label {
  font-weight: 500;
}

.tree-path {
  font-size: 12px;
  margin-left: 4px;
}

.tree-difficulty {
  flex-shrink: 0;
}
</style>
