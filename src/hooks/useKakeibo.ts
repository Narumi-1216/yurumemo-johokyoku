'use client'

import { useState, useCallback, useEffect } from 'react'
import { KakeiboData } from '@/types/kakeibo'
import { loadData, saveData } from '@/lib/kakeibo-storage'

export function useKakeibo() {
  const [data, setData] = useState<KakeiboData | null>(null)

  useEffect(() => {
    setData(loadData())
  }, [])

  const refresh = useCallback(() => {
    setData(loadData())
  }, [])

  const update = useCallback((updater: (d: KakeiboData) => KakeiboData) => {
    const current = loadData()
    const next = updater(current)
    saveData(next)
    setData(next)
  }, [])

  return { data, refresh, update }
}
