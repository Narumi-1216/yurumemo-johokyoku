export type TransactionType = 'expense' | 'income'

export type Category = {
  id: string
  name: string
  type: TransactionType
  color: string
  parentId: string | null
}

export type ClassificationRule = {
  id: string
  pattern: string
  categoryId: string
  priority: number
  matchType: 'contains' | 'startsWith' | 'exact'
}

export type Transaction = {
  id: string
  date: string
  amount: number
  vendorName: string
  categoryId: string | null
  memo: string
  source: string
  importedAt: string
}

export type KakeiboData = {
  transactions: Transaction[]
  categories: Category[]
  rules: ClassificationRule[]
  version: number
}

export const DEFAULT_CATEGORIES: Category[] = [
  { id: 'food', name: '食費', type: 'expense', color: '#E07B4A', parentId: null },
  { id: 'dining', name: '外食', type: 'expense', color: '#E0A04A', parentId: 'food' },
  { id: 'groceries', name: '食料品', type: 'expense', color: '#E0C44A', parentId: 'food' },
  { id: 'transport', name: '交通費', type: 'expense', color: '#4A7BE0', parentId: null },
  { id: 'utilities', name: '光熱費', type: 'expense', color: '#7B4AE0', parentId: null },
  { id: 'housing', name: '住居費', type: 'expense', color: '#4AE07B', parentId: null },
  { id: 'medical', name: '医療費', type: 'expense', color: '#E04A7B', parentId: null },
  { id: 'entertainment', name: '娯楽', type: 'expense', color: '#4AE0E0', parentId: null },
  { id: 'clothing', name: '衣類', type: 'expense', color: '#A04AE0', parentId: null },
  { id: 'education', name: '教育', type: 'expense', color: '#E0E04A', parentId: null },
  { id: 'communication', name: '通信費', type: 'expense', color: '#4AA0E0', parentId: null },
  { id: 'insurance', name: '保険', type: 'expense', color: '#E04AA0', parentId: null },
  { id: 'salary', name: '給与', type: 'income', color: '#3B7A57', parentId: null },
  { id: 'other_income', name: 'その他収入', type: 'income', color: '#5A9E75', parentId: null },
  { id: 'other_expense', name: 'その他支出', type: 'expense', color: '#AAAAAA', parentId: null },
]
