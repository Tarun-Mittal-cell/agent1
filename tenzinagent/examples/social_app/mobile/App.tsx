"""
React Native mobile app with elegant UI and animations
"""
import React from 'react'
import { NavigationContainer } from '@react-navigation/native'
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import { StatusBar } from 'expo-status-bar'
import { ThemeProvider } from '@shopify/restyle'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { theme } from './theme'
import { HomeScreen } from './screens/HomeScreen'
import { ProfileScreen } from './screens/ProfileScreen'
import { NotificationsScreen } from './screens/NotificationsScreen'
import { TabBar } from './components/TabBar'

const Tab = createBottomTabNavigator()
const queryClient = new QueryClient()

export default function App() {
  return (
    <SafeAreaProvider>
      <ThemeProvider theme={theme}>
        <QueryClientProvider client={queryClient}>
          <NavigationContainer>
            <StatusBar style="auto" />
            
            <Tab.Navigator
              tabBar={props => <TabBar {...props} />}
              screenOptions={{
                headerShown: false
              }}
            >
              <Tab.Screen
                name="Home"
                component={HomeScreen}
                options={{
                  tabBarIcon: 'home'
                }}
              />
              <Tab.Screen
                name="Profile"
                component={ProfileScreen}
                options={{
                  tabBarIcon: 'user'
                }}
              />
              <Tab.Screen
                name="Notifications"
                component={NotificationsScreen}
                options={{
                  tabBarIcon: 'bell'
                }}
              />
            </Tab.Navigator>
          </NavigationContainer>
        </QueryClientProvider>
      </ThemeProvider>
    </SafeAreaProvider>
  )
}