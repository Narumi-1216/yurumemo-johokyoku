'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const links = [
  { href: '/kakeibo', label: 'ダッシュボード' },
  { href: '/kakeibo/transactions', label: '取引一覧' },
  { href: '/kakeibo/rules', label: 'ルール管理' },
  { href: '/kakeibo/categories', label: 'カテゴリ' },
  { href: '/kakeibo/analytics', label: '分析' },
]

export default function KakeiboNav() {
  const pathname = usePathname()
  return (
    <nav className="flex gap-1 flex-wrap mb-8">
      {links.map((l) => {
        const active = l.href === '/kakeibo' ? pathname === l.href : pathname.startsWith(l.href)
        return (
          <Link
            key={l.href}
            href={l.href}
            className="px-3 py-1.5 rounded text-sm font-medium transition-colors"
            style={{
              background: active ? '#3B7A57' : '#EAF2EE',
              color: active ? '#fff' : '#3B7A57',
            }}
          >
            {l.label}
          </Link>
        )
      })}
    </nav>
  )
}
