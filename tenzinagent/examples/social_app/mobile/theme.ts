"""
Elegant mobile app theme with dark mode support
"""
import { createTheme } from '@shopify/restyle'

const palette = {
  purple: '#7928CA',
  blue: '#0070F3',
  black: '#000000',
  white: '#FFFFFF',
  gray50: '#F9FAFB',
  gray100: '#F3F4F6',
  gray200: '#E5E7EB',
  gray300: '#D1D5DB',
  gray400: '#9CA3AF',
  gray500: '#6B7280',
  gray600: '#4B5563',
  gray700: '#374151',
  gray800: '#1F2937',
  gray900: '#111827',
}

export const theme = createTheme({
  colors: {
    mainBackground: palette.white,
    mainForeground: palette.black,
    primaryCardBackground: palette.gray50,
    secondaryCardBackground: palette.gray100,
    primaryText: palette.gray900,
    secondaryText: palette.gray500,
    accent: palette.purple,
    accentLight: palette.blue,
    
    // Dark mode
    darkMainBackground: palette.gray900,
    darkMainForeground: palette.white,
    darkPrimaryCardBackground: palette.gray800,
    darkSecondaryCardBackground: palette.gray700,
    darkPrimaryText: palette.white,
    darkSecondaryText: palette.gray300,
  },
  spacing: {
    xs: 4,
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadii: {
    xs: 4,
    s: 8,
    m: 12,
    l: 16,
    xl: 24,
    xxl: 32,
  },
  textVariants: {
    header: {
      fontSize: 32,
      fontWeight: 'bold',
      color: 'primaryText',
    },
    subheader: {
      fontSize: 24,
      fontWeight: '600',
      color: 'primaryText',
    },
    body: {
      fontSize: 16,
      color: 'primaryText',
    },
    caption: {
      fontSize: 14,
      color: 'secondaryText',
    },
  },
  breakpoints: {
    phone: 0,
    tablet: 768,
  },
})

export type Theme = typeof theme
export type ThemeColors = keyof typeof theme.colors