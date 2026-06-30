'use client'

import { useState, useMemo } from 'react'
import Link from 'next/link'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import CategoryBadge from '@/components/kakeibo/CategoryBadge'
import { useKakeibo } from '@/hooks/useKakeibo'
import { bulkChangeVendorCategory, deleteTransaction, updateTransaction } from '@/lib/kakeibo-storage'
import { Transaction } from '@/types/kakeibo'

export default function TransactionsPage() {
  const { data, refresh } = useKakeibo()
  const [search, setSearch] = useState('')
  const [filterCat, setFilterCat] = useState<string>('')
  const [filterMonth, setFilterMonth] = useState<string>('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [bulkVendor, setBulkVendor] = useState<string>('')
  const [bulkCatId, setBulkCatId] = useState<string>('')
  const [bulkMsg, setBulkMsg] = useState<string | null>(null)

  const filtered = useMemo(() => {
    if (!data) return []
    return data.transactions.filter((t) => {
      if (search && !t.vendorName.toLowerCase().includes(search.toLowerCase()) && !t.memo.toLowerCase().includes(search.toLowerCase())) return false
      if (filterCat === '__unclassified') { if (t.categoryId !== null) return false }
      else if (filterCat && t.categoryId !== filterCat) return false
      if (filterMonth && !t.date.startsWith(filterMonth)) return false
      return true
    })
  }, [data, search, filterCat, filterMonth])

  const months = useMemo(() => {
    if (!data) return []
    const s = new Set(data.transactions.map((t) => t.date.slice(0, 7)))
    return Array.from(s).sort((a, b) => b.localeCompare(a))
  }, [data])

  function handleDelete(id: string) {
    if (!confirm('この取引を削除しますか？')) return
    deleteTransaction(id)
    refresh()
  }

  function handleCategoryChange(t: Transaction, catId: string) {
    updateTransaction({ ...t, categoryId: catId || null })
    refresh()
    setEditingId(null)
  }

  function handleBulkChange() {
    if (!bulkVendor || !bulkCatId) return
    const count = bulkChangeVendorCategory(bulkVendor, bulkCatId)
    setBulkMsg(`「${bulkVendor}」の${count}件のカテゴリを変更しました`)
    refresh()
    setBulkVendor('')
    setBulkCatId('')
    setTimeout(() => setBulkMsg(null), 4000)
  }

  if (!data) return <div className="py-12 text-center" style={{ color: '#888' }}>読み込み中...</div>

  const distinctVendors = Array.from(new Set(data.transactions.map((t) => t.vendorName))).sort()

  return (
    <div>
      <KakeiboNav />

      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">取引一覧</h1>
        <Link
          href="/kakeibo/transactions/import"
          className="text-sm px-3 py-1.5 rounded font-medium"
          style={{ background: '#3B7A57', color: '#fff' }}
        >
          + CSVインポート
        </Link>
      </div>

      {/* 一括カテゴリ変更 */}
      <div className="mb-6 p-4 rounded-lg" style={{ background: '#F7F4EF', border: '1px solid #E0DDD8' }}>
        <div className="text-sm font-medium mb-3">取引先名で一括カテゴリ変更</div>
        <div className="flex gap-2 items-end flex-wrap">
          <div>
            <label className="text-xs block mb-1" style={{ color: '#888' }}>取引先</label>
            <select
              value={bulkVendor}
              onChange={(e) => setBulkVendor(e.target.value)}
              className="text-sm border rounded px-2 py-1.5 min-w-48"
              style={{ borderColor: '#E0DDD8', background: '#fff' }}
            >
              <option value="">取引先を選択...</option>
              {distinctVendors.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs block mb-1" style={{ color: '#888' }}>変更先カテゴリ</label>
            <select
              value={bulkCatId}
              onChange={(e) => setBulkCatId(e.target.value)}
              className="text-sm border rounded px-2 py-1.5 min-w-40"
              style={{ borderColor: '#E0DDD8', background: '#fff' }}
            >
              <option value="">カテゴリを選択...</option>
              {data.categories.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleBulkChange}
            disabled={!bulkVendor || !bulkCatId}
            className="text-sm px-3 py-1.5 rounded font-medium disabled:opacity-40"
            style={{ background: '#3B7A57', color: '#fff' }}
          >
            一括変更
          </button>
        </div>
        {bulkMsg && (
          <div className="mt-2 text-sm" style={{ color: '#3B7A57' }}>{bulkMsg}</div>
        )}
      </div>

      {/* フィルター */}
      <div className="flex gap-2 mb-4 flex-wrap">
        <input
          type="text"
          placeholder="取引先・メモで検索"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="text-sm border rounded px-3 py-1.5 flex-1 min-w-40"
          style={{ borderColor: '#E0DDD8' }}
        />
        <select
          value={filterMonth}
          onChange={(e) => setFilterMonth(e.target.value)}
          className="text-sm border rounded px-2 py-1.5"
          style={{ borderColor: '#E0DDD8', background: '#fff' }}
        >
          <option value="">すべての月</option>
          {months.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <select
          value={filterCat}
          onChange={(e) => setFilterCat(e.target.value)}
          className="text-sm border rounded px-2 py-1.5"
          style={{ borderColor: '#E0DDD8', background: '#fff' }}
        >
          <option value="">すべてのカテゴリ</option>
          <option value="__unclassified">未分類のみ</option>
          {data.categories.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>
      </div>

      <div className="text-xs mb-3" style={{ color: '#888' }}>{filtered.length}件表示</div>

      <div className="divide-y" style={{ borderColor: '#E0DDD8' }}>
        {filtered.length === 0 && (
          <p className="text-sm py-8 text-center" style={{ color: '#888' }}>
            条件に一致する取引がありません
          </p>
        )}
        {filtered.map((t) => {
          const cat = data.categories.find((c) => c.id === t.categoryId)
          return (
            <div key={t.id} className="py-3">
              <div className="flex items-center gap-3">
                <div className="text-xs w-20 shrink-0" style={{ color: '#AAAAAA' }}>{t.date}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{t.vendorName}</div>
                  {t.memo && <div className="text-xs truncate" style={{ color: '#888' }}>{t.memo}</div>}
                </div>
                {editingId === t.id ? (
                  <select
                    defaultValue={t.categoryId ?? ''}
                    onChange={(e) => handleCategoryChange(t, e.target.value)}
                    autoFocus
                    onBlur={() => setEditingId(null)}
                    className="text-xs border rounded px-1 py-0.5"
                    style={{ borderColor: '#E0DDD8' }}
                  >
                    <option value="">未分類</option>
                    {data.categories.map((c) => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                ) : (
                  <button onClick={() => setEditingId(t.id)}>
                    <CategoryBadge category={cat} small />
                  </button>
                )}
                <div className="text-sm font-medium w-24 text-right shrink-0" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {Math.abs(t.amount).toLocaleString()}円
                </div>
                <button
                  onClick={() => handleDelete(t.id)}
                  className="text-xs shrink-0"
                  style={{ color: '#AAAAAA' }}
                >
                  削除
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
