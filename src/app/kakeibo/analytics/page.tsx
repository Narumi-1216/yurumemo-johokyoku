'use client'

import { useState, useMemo } from 'react'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import { useKakeibo } from '@/hooks/useKakeibo'
import {
  calcPeriodStats,
  buildMonthlyTrend,
  filterByPeriod,
  filterByYear,
  getAvailableYears,
} from '@/lib/kakeibo-stats'
import AmountDisplay from '@/components/kakeibo/AmountDisplay'
import CategoryBadge from '@/components/kakeibo/CategoryBadge'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts'

type Mode = 'period' | 'year'

export default function AnalyticsPage() {
  const { data } = useKakeibo()
  const [mode, setMode] = useState<Mode>('year')
  const [selectedYear, setSelectedYear] = useState<number>(new Date().getFullYear())
  const [fromDate, setFromDate] = useState<string>(() => {
    const d = new Date()
    d.setMonth(d.getMonth() - 5)
    return d.toISOString().slice(0, 7)
  })
  const [toDate, setToDate] = useState<string>(() => new Date().toISOString().slice(0, 7))
  const [trendCategoryIds, setTrendCategoryIds] = useState<string[]>([])

  const years = useMemo(() => {
    if (!data) return [new Date().getFullYear()]
    const available = getAvailableYears(data.transactions)
    return available.length > 0 ? available : [new Date().getFullYear()]
  }, [data])

  const filteredTx = useMemo(() => {
    if (!data) return []
    if (mode === 'year') return filterByYear(data.transactions, selectedYear)
    return filterByPeriod(data.transactions, fromDate + '-01', toDate + '-31')
  }, [data, mode, selectedYear, fromDate, toDate])

  const stats = useMemo(() => {
    if (!data) return null
    return calcPeriodStats(filteredTx, data.categories)
  }, [data, filteredTx])

  const trendFrom = mode === 'year' ? `${selectedYear}-01` : fromDate
  const trendTo = mode === 'year' ? `${selectedYear}-12` : toDate

  const trendData = useMemo(() => {
    if (!data || trendCategoryIds.length === 0) return []
    const points = buildMonthlyTrend(data.transactions, trendCategoryIds[0], trendFrom, trendTo)
    if (trendCategoryIds.length === 1) return points

    // merge multiple categories
    const extra = trendCategoryIds.slice(1).map((cid) =>
      buildMonthlyTrend(data.transactions, cid, trendFrom, trendTo)
    )
    return points.map((p, i) => {
      const row: Record<string, unknown> = { month: p.month, [trendCategoryIds[0]]: p.amount }
      for (let j = 0; j < extra.length; j++) {
        row[trendCategoryIds[j + 1]] = extra[j][i]?.amount ?? 0
      }
      return row
    })
  }, [data, trendCategoryIds, trendFrom, trendTo])

  function toggleCategory(id: string) {
    setTrendCategoryIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  if (!data) return <div className="py-12 text-center" style={{ color: '#888' }}>読み込み中...</div>

  const LINE_COLORS = ['#3B7A57', '#4A7BE0', '#E07B4A', '#7B4AE0', '#E04A7B']

  return (
    <div>
      <KakeiboNav />
      <h1 className="text-2xl font-bold mb-6">分析</h1>

      {/* 期間選択 */}
      <div className="mb-6 p-4 rounded-lg" style={{ background: '#F7F4EF', border: '1px solid #E0DDD8' }}>
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setMode('year')}
            className="text-sm px-3 py-1.5 rounded font-medium"
            style={{ background: mode === 'year' ? '#3B7A57' : '#EAF2EE', color: mode === 'year' ? '#fff' : '#3B7A57' }}
          >
            年別
          </button>
          <button
            onClick={() => setMode('period')}
            className="text-sm px-3 py-1.5 rounded font-medium"
            style={{ background: mode === 'period' ? '#3B7A57' : '#EAF2EE', color: mode === 'period' ? '#fff' : '#3B7A57' }}
          >
            任意期間
          </button>
        </div>

        {mode === 'year' ? (
          <div className="flex gap-2 flex-wrap">
            {years.map((y) => (
              <button
                key={y}
                onClick={() => setSelectedYear(y)}
                className="text-sm px-3 py-1 rounded border"
                style={{
                  background: selectedYear === y ? '#3B7A57' : '#fff',
                  color: selectedYear === y ? '#fff' : '#1A1A1A',
                  borderColor: selectedYear === y ? '#3B7A57' : '#E0DDD8',
                }}
              >
                {y}年
              </button>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <div>
              <label className="text-xs block mb-1" style={{ color: '#888' }}>開始月</label>
              <input
                type="month"
                value={fromDate}
                onChange={(e) => setFromDate(e.target.value)}
                className="text-sm border rounded px-2 py-1"
                style={{ borderColor: '#E0DDD8' }}
              />
            </div>
            <div className="mt-4 text-sm" style={{ color: '#888' }}>〜</div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#888' }}>終了月</label>
              <input
                type="month"
                value={toDate}
                onChange={(e) => setToDate(e.target.value)}
                className="text-sm border rounded px-2 py-1"
                style={{ borderColor: '#E0DDD8' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* サマリー */}
      {stats && (
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[
            { label: '合計支出', amount: stats.totalExpense, type: 'expense' as const, sub: `月平均 ${stats.avgExpense.toLocaleString()}円` },
            { label: '合計収入', amount: stats.totalIncome, type: 'income' as const, sub: `月平均 ${stats.avgIncome.toLocaleString()}円` },
            { label: '収支合計', amount: stats.net, type: 'net' as const, sub: `${stats.monthCount}ヶ月分` },
          ].map((c) => (
            <div key={c.label} className="rounded-lg p-4" style={{ background: '#fff', border: '1px solid #E0DDD8' }}>
              <div className="text-xs mb-1" style={{ color: '#888' }}>{c.label}</div>
              <div className="text-xl font-bold">
                <AmountDisplay amount={c.amount} type={c.type} />
              </div>
              <div className="text-xs mt-1" style={{ color: '#AAAAAA' }}>{c.sub}</div>
            </div>
          ))}
        </div>
      )}

      {/* カテゴリ別内訳 */}
      {stats && stats.byCategory.length > 0 && (
        <div className="mb-8">
          <h2 className="text-base font-semibold mb-4">カテゴリ別内訳</h2>
          <div className="space-y-2">
            {stats.byCategory.map(({ categoryId, total }) => {
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
                  <div className="text-sm w-28 text-right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {total.toLocaleString()}円
                  </div>
                  <div className="text-xs w-10 text-right" style={{ color: '#888' }}>{pct}%</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 推移グラフ */}
      <div className="mb-8">
        <h2 className="text-base font-semibold mb-3">支出推移（折れ線グラフ）</h2>
        <div className="mb-3">
          <p className="text-xs mb-2" style={{ color: '#888' }}>カテゴリを選択（複数可）</p>
          <div className="flex gap-2 flex-wrap">
            {data.categories
              .filter((c) => c.type === 'expense')
              .map((c, idx) => {
                const selected = trendCategoryIds.includes(c.id)
                return (
                  <button
                    key={c.id}
                    onClick={() => toggleCategory(c.id)}
                    className="text-xs px-2 py-1 rounded-full border transition-all"
                    style={{
                      background: selected ? c.color : '#fff',
                      color: selected ? '#fff' : c.color,
                      borderColor: c.color,
                    }}
                  >
                    {c.name}
                  </button>
                )
              })}
          </div>
        </div>

        {trendCategoryIds.length === 0 ? (
          <div className="h-48 flex items-center justify-center rounded-lg" style={{ background: '#F7F4EF', border: '1px solid #E0DDD8' }}>
            <p className="text-sm" style={{ color: '#888' }}>上からカテゴリを選んでください</p>
          </div>
        ) : trendData.length === 0 ? (
          <div className="h-48 flex items-center justify-center rounded-lg" style={{ background: '#F7F4EF', border: '1px solid #E0DDD8' }}>
            <p className="text-sm" style={{ color: '#888' }}>データがありません</p>
          </div>
        ) : (
          <div className="rounded-lg p-4" style={{ background: '#fff', border: '1px solid #E0DDD8' }}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={trendData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E0DDD8" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 11, fill: '#888' }}
                  tickFormatter={(v) => v.slice(5)}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#888' }}
                  tickFormatter={(v) => `${(v / 10000).toFixed(0)}万`}
                />
                <Tooltip
                  formatter={(value, name) => {
                    const cat = data.categories.find((c) => c.id === String(name))
                    return [`${Number(value).toLocaleString()}円`, cat?.name ?? String(name)]
                  }}
                  labelStyle={{ color: '#1A1A1A', fontWeight: 600 }}
                  contentStyle={{ border: '1px solid #E0DDD8', borderRadius: 6 }}
                />
                <Legend
                  formatter={(value) => {
                    const cat = data.categories.find((c) => c.id === value)
                    return cat?.name ?? value
                  }}
                />
                <ReferenceLine y={0} stroke="#E0DDD8" />
                {trendCategoryIds.map((cid, idx) => {
                  const cat = data.categories.find((c) => c.id === cid)
                  return (
                    <Line
                      key={cid}
                      type="monotone"
                      dataKey={cid}
                      stroke={cat?.color ?? LINE_COLORS[idx % LINE_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  )
                })}
              </LineChart>
            </ResponsiveContainer>

            {/* 前月比プラマイ表示 */}
            {trendCategoryIds.length === 1 && trendData.length >= 2 && (
              <div className="mt-4 pt-4" style={{ borderTop: '1px solid #E0DDD8' }}>
                <p className="text-xs font-medium mb-3" style={{ color: '#888' }}>月次推移（前月比）</p>
                <div className="flex gap-3 overflow-x-auto pb-1">
                  {(trendData as { month: string; amount?: number }[]).map((p, i) => {
                    const prev = i > 0 ? ((trendData[i - 1] as { amount?: number }).amount ?? 0) : null
                    const cur = p.amount ?? 0
                    const diff = prev !== null ? cur - prev : null
                    return (
                      <div key={p.month} className="shrink-0 text-center min-w-16">
                        <div className="text-xs mb-1" style={{ color: '#888' }}>{p.month.slice(5)}月</div>
                        <div className="text-sm font-medium" style={{ fontVariantNumeric: 'tabular-nums' }}>
                          {(cur / 10000).toFixed(1)}万
                        </div>
                        {diff !== null && (
                          <div
                            className="text-xs font-medium mt-0.5"
                            style={{ color: diff > 0 ? '#C0392B' : diff < 0 ? '#3B7A57' : '#888' }}
                          >
                            {diff > 0 ? `+${Math.round(diff / 1000)}k` : diff < 0 ? `${Math.round(diff / 1000)}k` : '±0'}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
