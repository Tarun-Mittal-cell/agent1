"""
Modern social media app home page with elegant UI
"""
import { Suspense } from 'react'
import { PostFeed } from '@/components/PostFeed'
import { Sidebar } from '@/components/Sidebar'
import { LoadingFeed } from '@/components/LoadingFeed'

export default function HomePage() {
  return (
    <main className="flex min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Sidebar */}
      <Sidebar className="w-64 hidden lg:block" />
      
      {/* Main Content */}
      <div className="flex-1 max-w-2xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500 mb-8">
          Your Feed
        </h1>
        
        {/* Posts Feed with Loading State */}
        <Suspense fallback={<LoadingFeed />}>
          <PostFeed />
        </Suspense>
      </div>
      
      {/* Right Sidebar - Trending */}
      <div className="w-80 hidden xl:block p-6">
        <div className="sticky top-6">
          <h2 className="text-xl font-semibold mb-4">Trending</h2>
          {/* Trending content */}
        </div>
      </div>
    </main>
  )
}