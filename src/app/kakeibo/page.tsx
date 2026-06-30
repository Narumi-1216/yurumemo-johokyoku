'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import AmountDisplay from '@/components/kakeibo/AmountDisplay'
import CategoryBadge from '@/components/kakeibo/CategoryBadge'
import { useKakeibo } from '@/hooks/useKakeibo'
import { calcPeriodStats, filterByYear, getAvailableYears } from '@/lib/kakeibo-stats'
import { bulkClassifyUnclassified } from '@/lib/kakeibo-storage'

export default function KakeiboPage() {
  const { data, refresh } = useKakeibo()
  const [selectedYear, setSelectedYear] = useState<number>(new Date().getFullYear())
  const [bulkResult, setBulkResult] = useState<string | null>(null)

  const years = useMemo(() => {
    if (!data) return [new Date().getFullYear()]
    const available = getAvailableYears(data.transactions)
    return available.length > 0 ? available : [new Date().getFullYear()]
  }, [data])

  const stats = useMemo(() => {
    if (!data) return null
    const filtered = filterByYear(data.transactions, selectedYear)
    return calcPeriodStats(filtered, data.categories)
  }, [data, selectedYear])

  const unclassifiedCount = useMemo(() => {
    if (!data) return 0
    return data.transactions.filter((t) => t.categoryId === null).length
  }, [data])

  const recentTransactions = useMemo(() => {
    if (!data) return []
    return data.transactions.slice(0, 10)
  }, [data])

  function handleBulkClassify() {
    const count = bulkClassifyUnclassified()
    setBulkResult(`${count}件の取引を自動分類しました`)
    refresh()
    setTimeout(() => setBulkResult(null), 4000)
  }

  if (!data) return <div className="py-12 text-center" style={{ color: '#888' }}>読み込み中...</div>

  return (
    <div>
      <KakeiboNav />

      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">ダッシュボード</h1>
        <div className="flex items-center gap-3">
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="text-sm border rounded px-2 py-1"
            style={{ borderColor: '#E0DDD8', background: '#fff' }}
          >
            {years.map((y) => (
              <option key={y} value={y}>{y}年</option>
            ))}
          </select>
          <Link
            href="/kakeibo/transactions/import"
            className="text-sm px-3 py-1.5 rounded font-medium"
            style={{ background: '#3B7A57', color: '#fff' }}
          >
            + CSVインポート
          </Link>
        </div>
      </div>

      {/* 一括分類バナー */}
      {unclassifiedCount > 0 && (
        <div
          className="mb-6 p-4 rounded-lg flex items-center justify-between"
          style={{ background: '#FFF8E7', border: '1px solid #F0C040' }}
        >
          <div>
            <span className="font-semibold" style={{ color: '#B8860B' }}>
              未分類の取引が{unclassifiedCount}件あります
            </span>
            {data.rules.length > 0 && (
              <span className="text-sm ml-2" style={{ color: '#888' }}>
                ルールに基づいて一括分類できます
              </span>
            )}
          </div>
          <div className="flex gap-2">
            {data.rules.length > 0 && (
              <button
                onClick={handleBulkClassify}
                className="text-sm px-3 py-1 rounded font-medium"
                style={{ background: '#B8860B', color: '#fff' }}
              >
                一括自動分類
              </button>
            )}
            <Link
              href="/kakeibo/rules"
              className="text-sm px-3 py-1 rounded"
              style={{ background: '#EAF2EE', color: '#3B7A57' }}
            >
              ルール設定
            </Link>
          </div>
        </div>
      )}

      {bulkResult && (
        <div
          className="mb-4 p-3 rounded text-sm font-medium"
          style={{ background: '#EAF2EE', color: '#3B7A57' }}
        >
          {bulkResult}
        </div>
      )}

      {/* サマリーカード */}
      {stats && (
        <div className="grid grid-cols-3 gap-4 mb-8">
          <SummaryCard label="年間支出" amount={stats.totalExpense} type="expense" sub={`月平均 ${stats.avgExpense.toLocaleString()}円`} />
          <SummaryCard label="年間収入" amount={stats.totalIncome} type="income" sub={`月平均 ${stats.avgIncome.toLocaleString()}円`} />
          <SummaryCard label="収支" amount={stats.net} type="net" sub={stats.net >= 0 ? '黒字' : '赤字'} />
        </div>
      )}

      {/* カテゴリ別支出 */}
      {stats && stats.byCategory.length > 0 && (
        <div className="mb-8">
          <h2 className="text-base font-semibold mb-3">カテゴリ別支出</h2>
          <div className="space-y-2">
            {stats.byCategory.slice(0, 8).map(({ categoryId, total }) => {
              const cat = data.categories.find((c) => c.id === categoryId)
              const pct = stats.totalExpense > 0 ? Math.round((total / stats.totalExpense) * 100) : 0
              return (
                <div key={categoryId} className="flex items-center gap-3">
                  <div className="w-28 shrink-0">
                    <CategoryBadge category={cat} small />
                  </div>
                  <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: '#E0DDD8' }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, background: cat?.color ?? '#AAAAAA' }}
                    />
                  </div>
                  <div className="text-sm w-24 text-right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {total.toLocaleString()}円
                  </div>
                  <div className="text-xs w-8 text-right" style={{ color: '#888' }}>{pct}%</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 直近の取引 */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold">直近の取引</h2>
          <Link href="/kakeibo/transactions" className="text-sm" style={{ color: '#3B7A57' }}>
            すべて見る →
          </Link>
        </div>
        {recentTransactions.length === 0 ? (
          <p className="text-sm py-8 text-center" style={{ color: '#888' }}>
            取引データがありません。CSVをインポートするか手動で追加してください。
          </p>
        ) : (
          <div className="divide-y" style={{ borderColor: '#E0DDD8' }}>
            {recentTransactions.map((t) => {
              const cat = data.categories.find((c) => c.id === t.categoryId)
              return (
                <div key={t.id} className="py-3 flex items-center gap-3">
                  <div className="text-xs w-20 shrink-0" style={{ color: '#AAAAAA' }}>{t.date}</div>
                  <div className="flex-1 text-sm truncate">{t.vendorName}</div>
                  <CategoryBadge category={cat} small />
                  <div className="text-sm font-medium w-24 text-right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {Math.abs(t.amount).toLocaleString()}円
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function SummaryCard({ label, amount, type, sub }: { label: string; amount: number; type: 'expense' | 'income' | 'net'; sub: string }) {
  return (
    <div className="rounded-lg p-4" style={{ background: '#fff', border: '1px solid #E0DDD8' }}>
      <div className="text-xs mb-1" style={{ color: '#888' }}>{label}</div>
      <div className="text-xl font-bold">
        <AmountDisplay amount={amount} type={type} />
      </div>
      <div className="text-xs mt-1" style={{ color: '#AAAAAA' }}>{sub}</div>
    </div>
  )
}
