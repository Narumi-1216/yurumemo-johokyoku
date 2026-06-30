import { Transaction } from '@/types/kakeibo'
import { nanoid } from './nanoid'

export type CsvColumn = {
  date: number
  vendorName: number
  amount: number
  memo?: number
}

/** 汎用CSVパーサー（ヘッダー行あり想定） */
export function parseCsv(
  text: string,
  mapping: CsvColumn,
  source: string,
): Transaction[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)

  if (lines.length < 2) return []

  const rows = lines.slice(1).map((line) => splitCsvLine(line))
  const now = new Date().toISOString()

  return rows
    .map((cols): Transaction | null => {
      const date = cols[mapping.date]?.trim() ?? ''
      const vendorName = cols[mapping.vendorName]?.trim() ?? ''
      const rawAmount = cols[mapping.amount]?.trim().replace(/,/g, '') ?? ''
      const amount = parseFloat(rawAmount)
      if (!date || !vendorName || isNaN(amount)) return null
      const normalizedDate = normalizeDate(date)
      if (!normalizedDate) return null
      return {
        id: nanoid(),
        date: normalizedDate,
        amount,
        vendorName,
        categoryId: null,
        memo: mapping.memo !== undefined ? (cols[mapping.memo]?.trim() ?? '') : '',
        source,
        importedAt: now,
      }
    })
    .filter((t): t is Transaction => t !== null)
}

function splitCsvLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuote = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') {
      inQuote = !inQuote
    } else if (ch === ',' && !inQuote) {
      result.push(current)
      current = ''
    } else {
      current += ch
    }
  }
  result.push(current)
  return result
}

function normalizeDate(raw: string): string | null {
  // YYYY/MM/DD → YYYY-MM-DD
  const m1 = raw.match(/^(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})$/)
  if (m1) return `${m1[1]}-${m1[2].padStart(2, '0')}-${m1[3].padStart(2, '0')}`
  // YYYYMMDD
  const m2 = raw.match(/^(\d{4})(\d{2})(\d{2})$/)
  if (m2) return `${m2[1]}-${m2[2]}-${m2[3]}`
  return null
}
