'use client'

import { useState } from 'react'
import KakeiboNav from '@/components/kakeibo/KakeiboNav'
import { useKakeibo } from '@/hooks/useKakeibo'
import { upsertRule, deleteRule, bulkClassifyUnclassified } from '@/lib/kakeibo-storage'
import { ClassificationRule } from '@/types/kakeibo'
import { nanoid } from '@/lib/nanoid'

export default function RulesPage() {
  const { data, refresh } = useKakeibo()
  const [editing, setEditing] = useState<ClassificationRule | null>(null)
  const [bulkMsg, setBulkMsg] = useState<string | null>(null)

  const unclassifiedCount = data?.transactions.filter((t) => t.categoryId === null).length ?? 0

  function handleSave() {
    if (!editing) return
    if (!editing.pattern || !editing.categoryId) return
    upsertRule(editing)
    setEditing(null)
    refresh()
  }

  function handleDelete(id: string) {
    if (!confirm('このルールを削除しますか？')) return
    deleteRule(id)
    refresh()
  }

  function handleBulkClassify() {
    const count = bulkClassifyUnclassified()
    setBulkMsg(`${count}件の未分類取引を自動分類しました`)
    refresh()
    setTimeout(() => setBulkMsg(null), 4000)
  }

  function newRule(): ClassificationRule {
    return { id: nanoid(), pattern: '', categoryId: '', priority: 10, matchType: 'contains' }
  }

  if (!data) return <div className="py-12 text-center" style={{ color: '#888' }}>読み込み中...</div>

  return (
    <div>
      <KakeiboNav />

      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">分類ルール管理</h1>
        <div className="flex gap-2">
          {unclassifiedCount > 0 && (
            <button
              onClick={handleBulkClassify}
              className="text-sm px-3 py-1.5 rounded font-medium"
              style={{ background: '#B8860B', color: '#fff' }}
            >
              未分類{unclassifiedCount}件を一括分類
            </button>
          )}
          <button
            onClick={() => setEditing(newRule())}
            className="text-sm px-3 py-1.5 rounded font-medium"
            style={{ background: '#3B7A57', color: '#fff' }}
          >
            + ルール追加
          </button>
        </div>
      </div>

      {bulkMsg && (
        <div className="mb-4 p-3 rounded text-sm font-medium" style={{ background: '#EAF2EE', color: '#3B7A57' }}>
          {bulkMsg}
        </div>
      )}

      <div className="mb-4 p-4 rounded-lg text-sm" style={{ background: '#F7F4EF', border: '1px solid #E0DDD8' }}>
        <p className="font-medium mb-1">ルールの動作</p>
        <ul className="list-disc list-inside space-y-1 text-xs" style={{ color: '#666' }}>
          <li>優先度の高いルールから順に評価されます</li>
          <li>最初にマッチしたルールのカテゴリが適用されます</li>
          <li>「一括分類」は未分類の取引のみに適用されます（分類済みは変更されません）</li>
        </ul>
      </div>

      {/* ルール編集フォーム */}
      {editing && (
        <div className="mb-6 p-4 rounded-lg" style={{ background: '#EAF2EE', border: '1px solid #3B7A5733' }}>
          <div className="text-sm font-medium mb-4">
            {data.rules.find((r) => r.id === editing.id) ? 'ルール編集' : 'ルール追加'}
          </div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>取引先パターン</label>
              <input
                type="text"
                value={editing.pattern}
                onChange={(e) => setEditing({ ...editing, pattern: e.target.value })}
                placeholder="例: スターバックス"
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8' }}
              />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>マッチ方法</label>
              <select
                value={editing.matchType}
                onChange={(e) => setEditing({ ...editing, matchType: e.target.value as ClassificationRule['matchType'] })}
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8', background: '#fff' }}
              >
                <option value="contains">部分一致（含む）</option>
                <option value="startsWith">前方一致（始まる）</option>
                <option value="exact">完全一致</option>
              </select>
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>カテゴリ</label>
              <select
                value={editing.categoryId}
                onChange={(e) => setEditing({ ...editing, categoryId: e.target.value })}
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8', background: '#fff' }}
              >
                <option value="">選択してください</option>
                {data.categories.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: '#666' }}>優先度（高いほど先に評価）</label>
              <input
                type="number"
                value={editing.priority}
                onChange={(e) => setEditing({ ...editing, priority: Number(e.target.value) })}
                className="w-full text-sm border rounded px-2 py-1.5"
                style={{ borderColor: '#E0DDD8' }}
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!editing.pattern || !editing.categoryId}
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

      {/* ルール一覧 */}
      {data.rules.length === 0 ? (
        <p className="text-sm py-8 text-center" style={{ color: '#888' }}>
          ルールがありません。「+ ルール追加」からルールを作成してください。
        </p>
      ) : (
        <div className="divide-y" style={{ borderColor: '#E0DDD8' }}>
          {data.rules.map((rule) => {
            const cat = data.categories.find((c) => c.id === rule.categoryId)
            return (
              <div key={rule.id} className="py-3 flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{rule.pattern}</div>
                  <div className="text-xs mt-0.5" style={{ color: '#888' }}>
                    {rule.matchType === 'contains' ? '部分一致' : rule.matchType === 'startsWith' ? '前方一致' : '完全一致'}
                    　→　{cat?.name ?? '不明'}　（優先度: {rule.priority}）
                  </div>
                </div>
                <button
                  onClick={() => setEditing({ ...rule })}
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: '#EAF2EE', color: '#3B7A57' }}
                >
                  編集
                </button>
                <button
                  onClick={() => handleDelete(rule.id)}
                  className="text-xs"
                  style={{ color: '#AAAAAA' }}
                >
                  削除
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
