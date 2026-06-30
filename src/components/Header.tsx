import Link from 'next/link'

export default function Header() {
  return (
    <header>
      {/* アクセントライン（ページ上端の細いグリーンライン） */}
      <div style={{ height: '3px', background: '#3B7A57' }} />

      <div className="max-w-2xl mx-auto px-6 py-6 flex items-baseline justify-between">
        <Link
          href="/"
          className="text-lg font-bold tracking-tight"
          style={{ color: '#1A1A1A' }}
        >
          ゆるめも情報局
        </Link>

        <nav className="flex gap-8 text-sm" style={{ color: '#888888' }}>
          <Link
            href="/"
            className="transition-colors hover:text-[#3B7A57]"
          >
            ホーム
          </Link>
          <Link
            href="/about"
            className="transition-colors hover:text-[#3B7A57]"
          >
            このブログについて
          </Link>
          <Link
            href="/kakeibo"
            className="transition-colors hover:text-[#3B7A57]"
          >
            家計管理
          </Link>
        </nav>
      </div>
    </header>
  )
}
