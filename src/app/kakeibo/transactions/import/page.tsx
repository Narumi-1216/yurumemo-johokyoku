'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import { parseCsv, CsvColumn } from '@/lib/kakeibo-csv'
import { addTransactions } from '@/lib/kakeibo-storage'

const PRESETS: { label: string; mapping: CsvColumn; note: string }[] = [
  {
    label: '汎用（列番号指定）',
    mapping: { date: 0, vendorName: 1, amount: 2, memo: 3 },
    note: '日付・取引先・金額・メモの列番号を手動で設定してください',
  },
  {
    label: '楽天カード',
    mapping: { date: 0, vendorName: 1, amount: 5, memo: 2 },
    note: '利用日,利用店名等,利用者,支払方法,利用金額,支払金額,...',
  },
  {
    label: 'イオン銀行',
    mapping: { date: 0, vendorName: 1, amount: 3, memo: 4 },
    note: '年月日,お取引内容,お支払い金額,お預け入れ金額,残高,...',
  },
]

export default function ImportPage() {
  const router = useRouter()
  const fileRef = useRef<HTMLInputElement>(null)
  const [csvText, setCsvText] = useState('')
  const [previewLines, setPreviewLines] = useState<string[]>([])
  const [preset, setPreset] = useState(0)
  const [mapping, setMapping] = useState<CsvColumn>({ date: 0, vendorName: 1, amount: 2, memo: 3 })
  const [source, setSource] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setSource(file.name)
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result as string
      setCsvText(text)
      const lines = text.split(/\r?\n/).filter(Boolean)
      setPreviewLines(lines.slice(0, 5))
    }
    reader.readAsText(file, 'UTF-8')
  }

  function handlePreset(idx: number) {
    setPreset(idx)
    setMapping({ ...PRESETS[idx].mapping })
  }

  function handleImport() {
    if (!csvText) { setError('CSVファイルを選択してください'); return }
    try {
      const transactions = parseCsv(csvText, mapping, source || 'manual')
      if (transactions.length === 0) { setError('取引データが見つかりませんでした。列の設定を確認してください'); return }
      addTransactions(transactions)
      setResult(`${transactions.length}件の取引をインポートしました`)
      setError(null)
      setTimeout(() => router.push('/kakeibo/transactions'), 1500)
    } catch (e) {
      setError('CSVの解析に失敗しました: ' + String(e))
    }
  }

  return (
    <div>
      <KakeiboNav />
      <h1 className="text-2xl font-bold mb-6">CSVインポート</h1>

      <div className="space-y-6">
        {/* ファイル選択 */}
        <div>
          <label className="text-sm font-medium block mb-2">CSVファイル</label>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.txt"
            onChange={handleFile}
            className="text-sm"
          />
        </div>

        {/* プリセット */}
        <div>
          <label className="text-sm font-medium block mb-2">フォーマット</label>
          <div className="flex gap-2 flex-wrap mb-2">
            {PRESETS.map((p, i) => (
              <button
                key={i}
                onClick={() => handlePreset(i)}
                className="text-sm px-3 py-1 rounded"
                style={{
                  background: preset === i ? '#3B7A57' : '#EAF2EE',
                  color: preset === i ? '#fff' : '#3B7A57',
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <p className="text-xs" style={{ color: '#888' }}>{PRESETS[preset].note}</p>
        </div>

        {/* 列マッピング */}
        <div>
          <label className="text-sm font-medium block mb-2">列番号設定（0始まり）</label>
          <div className="flex gap-4 flex-wrap">
            {(
              [
                { key: 'date', label: '日付列' },
                { key: 'vendorName', label: '取引先列' },
                { key: 'amount', label: '金額列' },
                { key: 'memo', label: 'メモ列（省略可）' },
              ] as { key: keyof CsvColumn; label: string }[]
            ).map(({ key, label }) => (
              <div key={key}>
                <label className="text-xs block mb-1" style={{ color: '#888' }}>{label}</label>
                <input
                  type="number"
                  min={0}
                  value={mapping[key] ?? ''}
                  onChange={(e) => setMapping((m) => ({ ...m, [key]: e.target.value === '' ? undefined : Number(e.target.value) }))}
                  className="w-20 text-sm border rounded px-2 py-1"
                  style={{ borderColor: '#E0DDD8' }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* プレビュー */}
        {previewLines.length > 0 && (
          <div>
            <label className="text-sm font-medium block mb-2">プレビュー（先頭5行）</label>
            <div className="overflow-x-auto rounded border text-xs p-3 font-mono" style={{ background: '#F7F4EF', borderColor: '#E0DDD8' }}>
              {previewLines.map((line, i) => (
                <div key={i} style={{ color: i === 0 ? '#888' : '#1A1A1A' }}>{line}</div>
              ))}
            </div>
          </div>
        )}

        {/* 取り込みボタン */}
        <div>
          {error && <div className="mb-3 text-sm" style={{ color: '#C0392B' }}>{error}</div>}
          {result && <div className="mb-3 text-sm font-medium" style={{ color: '#3B7A57' }}>{result}</div>}
          <button
            onClick={handleImport}
            className="px-4 py-2 rounded font-medium"
            style={{ background: '#3B7A57', color: '#fff' }}
          >
            インポート実行
          </button>
        </div>
      </div>
    </div>
  )
}
