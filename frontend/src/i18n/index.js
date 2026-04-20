import { createI18n } from 'vue-i18n'
import languages from '../../../locales/languages.json'

const localeFiles = import.meta.glob('../../../locales/!(languages).json', { eager: true })

const rawMessages = {}
const localeAliases = {
  'zh-cn': 'zh',
  'zh-hans': 'zh',
  'zh-hant': 'zh',
  cn: 'zh',
  china: 'zh',
  'en-us': 'en',
  'en-gb': 'en',
  'vi-vn': 'vi'
}

const normalizeLocale = (value) => {
  if (!value) return 'zh'
  const token = String(value).split(',')[0].trim().toLowerCase().split(';')[0].trim()
  if (localeAliases[token]) return localeAliases[token]
  const base = token.split('-')[0]
  if (localeAliases[base]) return localeAliases[base]
  if (languages[base]) return base
  return token
}

for (const path in localeFiles) {
  const key = path.match(/\/([^/]+)\.json$/)[1]
  rawMessages[key] = localeFiles[path].default
}

const fallbackMessages = rawMessages.zh || {}
const messages = {}
const availableLocales = []

for (const [key, meta] of Object.entries(languages)) {
  messages[key] = rawMessages[key] || fallbackMessages
  availableLocales.push({ key, label: meta.label })
}

const savedLocale = normalizeLocale(localStorage.getItem('locale') || 'zh')
localStorage.setItem('locale', savedLocale)

const i18n = createI18n({
  legacy: false,
  locale: savedLocale,
  fallbackLocale: 'zh',
  messages
})

export { availableLocales }
export default i18n
