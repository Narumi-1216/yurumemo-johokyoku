import { Category } from '@/types/kakeibo'

type Props = { category: Category | undefined; small?: boolean }

export default function CategoryBadge({ category, small }: Props) {
  if (!category) {
    return (
      <span
        className={`inline-block rounded-full px-2 py-0.5 ${small ? 'text-xs' : 'text-sm'}`}
        style={{ background: '#E0DDD8', color: '#888' }}
      >
        未分類
      </span>
    )
  }
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 font-medium ${small ? 'text-xs' : 'text-sm'}`}
      style={{ background: category.color + '22', color: category.color, border: `1px solid ${category.color}44` }}
    >
      {category.name}
    </span>
  )
}
