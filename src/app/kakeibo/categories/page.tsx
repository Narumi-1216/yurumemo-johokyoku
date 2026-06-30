'use client'

import { useState } from 'react'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import { useKakeibo } from '@/hooks/useKakeibo'
import { upsertCategory, deleteCategory } from '@/lib/kakeibo-storage'
import { Category, TransactionType } from '@/types/kakeibo'
import { nanoid } from '@/lib/nanoid'

const PALETTE = [
  '#E07B4A', '#E0A04A', '#E0C44A', '#3B7A57', '#4AE07B',
  '#4A7BE0', '#4AE0E0', '#7B4AE0', '#A04AE0', '#E04A7B',
  '#E04AA0', '#AAAAAA',
]

export default function CategoriesPage() {
  const { data, refresh } = useKakeibo()
  const [editing, setEditing] = useState<Category | null>(null)

  function newCategory(): Category {
    return { id: nanoid(), name: '', type: 'expense', color: '#3B7A57', parentId: null }
  }

  function handleSave() {
    if (!editing || !editing.name) return
    upsertCategory(editing)
    setEditing(null)
    refresh()
  }

  function handleDelete(id: string) {
    if (!confirm('このカテゴリを削除しますか？関連する取引の分類は「未分類」になります。')) return
    deleteCategory(id)
    refresh()
  }

  if (!data) return <div className="py-12 text-center" style={{ color: '#888' }}>読み込み中...</div>

  const expenseCategories = data.categories.filter((c) => c.type === 'expense' && !c.parentId)
  const incomeCategories = data.categories.filter((c) => c.type === 'income' && !c.parentId)
  const subCategories = data.categories.filter((c) => c.parentId)

  return (
    <div>
      <KakeiboNav />

      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">カテゴリ管理</h1>
        <button
          onClick={() => setEditing(newCategory())}
          className="text-sm px-3 py-1.5 rounded font-medium"
          style={{ background: '#3B7A57', color: '#fff' }}
        >
          + カテゴリ追加
        </button>
      </div>

      {editing && (
        <div className="mb-6 p-4 rounded-lg" style={{ background: '#EAF2EE', border: '1px solid #3B7A5733' }}>
          <div className="text-sm font-medium mb-4">
            {data.categories.find((c) => c.id === editing.id) ? 'カテゴリ編集' : 'カテゴリ追加'}
          </div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>カテゴリ名</label>
              <input
                type="text"
                value={editing.name}
                onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                placeholder="例: 食費"
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8' }}
              />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>種別</label>
              <select
                value={editing.type}
                onChange={(e) => setEditing({ ...editing, type: e.target.value as TransactionType })}
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8', background: '#fff' }}
              >
                <option value="expense">支出</option>
                <option value="income">収入</option>
              </select>
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>親カテゴリ（省略可）</label>
              <select
                value={editing.parentId ?? ''}
                onChange={(e) => setEditing({ ...editing, parentId: e.target.value || null })}
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8', background: '#fff' }}
              >
                <option value="">なし（トップレベル）</option>
                {data.categories
                  .filter((c) => !c.parentId && c.id !== editing.id)
                  .map((c) => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
              </select>
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>カラー</label>
              <div className="flex gap-1.5 flex-wrap">
                {PALETTE.map((color) => (
                  <button
                    key={color}
                    onClick={() => setEditing({ ...editing, color })}
                    className="w-6 h-6 rounded-full border-2 transition-all"
                    style={{
                      background: color,
                      borderColor: editing.color === color ? '#1A1A1A' : 'transparent',
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!editing.name}
              className="text-sm px-4 py-1.5 rounded font-medium disabled:opacity-40"
              style={{ background: '#3B7A57', color: '#fff' }}
            >
              保存
            </button>
            <button
              onClick={() => setEditing(null)}
              className="text-sm px-4 py-1.5 rounded"
              style={{ background: '#E0DDD8', color: '#666' }}
            >
              キャンセル
            </button>
          </div>
        </div>
      )}

      {[
        { label: '支出カテゴリ', cats: expenseCategories },
        { label: '収入カテゴリ', cats: incomeCategories },
      ].map(({ label, cats }) => (
        <div key={label} className="mb-6">
          <h2 className="text-sm font-semibold mb-3" style={{ color: '#888' }}>{label}</h2>
          <div className="divide-y" style={{ borderColor: '#E0DDD8' }}>
            {cats.map((cat) => {
              const children = subCategories.filter((c) => c.parentId === cat.id)
              const usageCount = data.transactions.filter((t) => t.categoryId === cat.id).length
              return (
                <div key={cat.id} className="py-3">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full shrink-0" style={{ background: cat.color }} />
                    <div className="flex-1 text-sm font-medium">{cat.name}</div>
                    <div className="text-xs" style={{ color: '#AAAAAA' }}>{usageCount}件</div>
                    <button
                      onClick={() => setEditing({ ...cat })}
                      className="text-xs px-2 py-0.5 rounded"
                      style={{ background: '#EAF2EE', color: '#3B7A57' }}
                    >
                      編集
                    </button>
                    <button
                      onClick={() => handleDelete(cat.id)}
                      className="text-xs"
                      style={{ color: '#AAAAAA' }}
                    >
                      削除
                    </button>
                  </div>
                  {children.map((child) => (
                    <div key={child.id} className="ml-6 mt-2 flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ background: child.color }} />
                      <div className="flex-1 text-xs">{child.name}</div>
                      <button
                        onClick={() => setEditing({ ...child })}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: '#EAF2EE', color: '#3B7A57' }}
                      >
                        編集
                      </button>
                      <button
                        onClick={() => handleDelete(child.id)}
                        className="text-xs"
                        style={{ color: '#AAAAAA' }}
                      >
                        削除
                      </button>
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
