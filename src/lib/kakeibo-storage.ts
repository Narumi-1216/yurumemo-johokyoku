'use client'

import { KakeiboData, Transaction, Category, ClassificationRule, DEFAULT_CATEGORIES } from '@/types/kakeibo'

const STORAGE_KEY = 'yurumemo_kakeibo_v1'

function emptyData(): KakeiboData {
  return {
    transactions: [],
    categories: DEFAULT_CATEGORIES,
    rules: [],
    version: 1,
  }
}

export function loadData(): KakeiboData {
  if (typeof window === 'undefined') return emptyData()
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return emptyData()
    return JSON.parse(raw) as KakeiboData
  } catch {
    return emptyData()
  }
}

export function saveData(data: KakeiboData): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

export function exportJson(): string {
  return JSON.stringify(loadData(), null, 2)
}

export function importJson(json: string): void {
  const data = JSON.parse(json) as KakeiboData
  saveData(data)
}

export function addTransactions(newItems: Transaction[]): void {
  const data = loadData()
  const existingIds = new Set(data.transactions.map((t) => t.id))
  for (const item of newItems) {
    if (!existingIds.has(item.id)) {
      data.transactions.push(item)
    }
  }
  data.transactions.sort((a, b) => b.date.localeCompare(a.date))
  saveData(data)
}

export function updateTransaction(updated: Transaction): void {
  const data = loadData()
  const idx = data.transactions.findIndex((t) => t.id === updated.id)
  if (idx !== -1) data.transactions[idx] = updated
  saveData(data)
}

export function deleteTransaction(id: string): void {
  const data = loadData()
  data.transactions = data.transactions.filter((t) => t.id !== id)
  saveData(data)
}

export function upsertCategory(cat: Category): void {
  const data = loadData()
  const idx = data.categories.findIndex((c) => c.id === cat.id)
  if (idx !== -1) data.categories[idx] = cat
  else data.categories.push(cat)
  saveData(data)
}

export function deleteCategory(id: string): void {
  const data = loadData()
  data.categories = data.categories.filter((c) => c.id !== id)
  saveData(data)
}

export function upsertRule(rule: ClassificationRule): void {
  const data = loadData()
  const idx = data.rules.findIndex((r) => r.id === rule.id)
  if (idx !== -1) data.rules[idx] = rule
  else data.rules.push(rule)
  data.rules.sort((a, b) => b.priority - a.priority)
  saveData(data)
}

export function deleteRule(id: string): void {
  const data = loadData()
  data.rules = data.rules.filter((r) => r.id !== id)
  saveData(data)
}

function matchesRule(vendorName: string, rule: ClassificationRule): boolean {
  const v = vendorName.toLowerCase()
  const p = rule.pattern.toLowerCase()
  if (rule.matchType === 'exact') return v === p
  if (rule.matchType === 'startsWith') return v.startsWith(p)
  return v.includes(p)
}

/** 未分類の取引をルールで一括分類。変更件数を返す */
export function bulkClassifyUnclassified(): number {
  const data = loadData()
  const sortedRules = [...data.rules].sort((a, b) => b.priority - a.priority)
  let count = 0
  for (const t of data.transactions) {
    if (t.categoryId !== null) continue
    for (const rule of sortedRules) {
      if (matchesRule(t.vendorName, rule)) {
        t.categoryId = rule.categoryId
        count++
        break
      }
    }
  }
  saveData(data)
  return count
}

/** 同じ取引先名の取引を指定カテゴリに一括変更。変更件数を返す */
export function bulkChangeVendorCategory(vendorName: string, categoryId: string): number {
  const data = loadData()
  let count = 0
  for (const t of data.transactions) {
    if (t.vendorName === vendorName && t.categoryId !== categoryId) {
      t.categoryId = categoryId
      count++
    }
  }
  saveData(data)
  return count
}

export function getDistinctVendors(): { vendorName: string; categoryId: string | null; count: number }[] {
  const data = loadData()
  const map = new Map<string, { categoryId: string | null; count: number }>()
  for (const t of data.transactions) {
    const existing = map.get(t.vendorName)
    if (!existing) {
      map.set(t.vendorName, { categoryId: t.categoryId, count: 1 })
    } else {
      existing.count++
      // most common category
      if (t.categoryId && existing.categoryId !== t.categoryId) {
        existing.categoryId = t.categoryId
      }
    }
  }
  return Array.from(map.entries())
    .map(([vendorName, v]) => ({ vendorName, ...v }))
    .sort((a, b) => b.count - a.count)
}
