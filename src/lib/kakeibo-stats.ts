import { Transaction, Category } from '@/types/kakeibo'

export type PeriodStats = {
  totalExpense: number
  totalIncome: number
  net: number
  byCategory: { categoryId: string; total: number }[]
  monthCount: number
  avgExpense: number
  avgIncome: number
}

export type MonthlyPoint = {
  month: string
  amount: number
  cumulative?: number
}

export function filterByPeriod(
  transactions: Transaction[],
  from: string,
  to: string,
): Transaction[] {
  return transactions.filter((t) => t.date >= from && t.date <= to)
}

export function filterByYear(transactions: Transaction[], year: number): Transaction[] {
  const prefix = String(year)
  return transactions.filter((t) => t.date.startsWith(prefix))
}

export function calcPeriodStats(
  transactions: Transaction[],
  categories: Category[],
): PeriodStats {
  const expenseTypes = new Set(
    categories.filter((c) => c.type === 'expense').map((c) => c.id),
  )
  const incomeTypes = new Set(
    categories.filter((c) => c.type === 'income').map((c) => c.id),
  )

  let totalExpense = 0
  let totalIncome = 0
  const byCat = new Map<string, number>()

  for (const t of transactions) {
    const catId = t.categoryId ?? 'uncategorized'
    byCat.set(catId, (byCat.get(catId) ?? 0) + Math.abs(t.amount))
    if (t.categoryId && incomeTypes.has(t.categoryId)) {
      totalIncome += Math.abs(t.amount)
    } else {
      totalExpense += Math.abs(t.amount)
    }
  }

  const months = new Set(transactions.map((t) => t.date.slice(0, 7)))
  const monthCount = months.size || 1

  return {
    totalExpense,
    totalIncome,
    net: totalIncome - totalExpense,
    byCategory: Array.from(byCat.entries())
      .map(([categoryId, total]) => ({ categoryId, total }))
      .sort((a, b) => b.total - a.total),
    monthCount,
    avgExpense: Math.round(totalExpense / monthCount),
    avgIncome: Math.round(totalIncome / monthCount),
  }
}

export function buildMonthlyTrend(
  transactions: Transaction[],
  categoryId: string,
  from: string,
  to: string,
): MonthlyPoint[] {
  const fromDate = new Date(from + '-01')
  const toDate = new Date(to + '-01')
  const points: MonthlyPoint[] = []

  let d = new Date(fromDate)
  while (d <= toDate) {
    const month = d.toISOString().slice(0, 7)
    const total = transactions
      .filter((t) => t.date.startsWith(month) && t.categoryId === categoryId)
      .reduce((sum, t) => sum + Math.abs(t.amount), 0)
    points.push({ month, amount: total })
    d.setMonth(d.getMonth() + 1)
  }

  // cumulative delta from first month
  const first = points[0]?.amount ?? 0
  let cum = 0
  for (const p of points) {
    cum += p.amount - first
    p.cumulative = cum
  }

  return points
}

export function getAvailableYears(transactions: Transaction[]): number[] {
  const years = new Set(transactions.map((t) => parseInt(t.date.slice(0, 4))))
  return Array.from(years).sort((a, b) => b - a)
}
