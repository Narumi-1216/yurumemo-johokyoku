import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: {
    default: '家計管理',
    template: '%s | 家計管理',
  },
}

export default function KakeiboLayout({ children }: { children: React.ReactNode }) {
  return (
    <div>
      {children}
    </div>
  )
}
