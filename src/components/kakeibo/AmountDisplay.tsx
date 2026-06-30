type Props = { amount: number; type?: 'expense' | 'income' | 'net' }

export default function AmountDisplay({ amount, type }: Props) {
  const isNegative = amount < 0
  const color =
    type === 'income'
      ? '#3B7A57'
      : type === 'net'
      ? amount >= 0
        ? '#3B7A57'
        : '#C0392B'
      : isNegative
      ? '#3B7A57'
      : '#C0392B'

  const prefix = type === 'net' ? (amount >= 0 ? '+' : '') : ''
  return (
    <span style={{ color, fontVariantNumeric: 'tabular-nums' }}>
      {prefix}
      {Math.abs(amount).toLocaleString('ja-JP')}円
    </span>
  )
}
