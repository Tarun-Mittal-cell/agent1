"""
Elegant post feed component with animations and interactions
"""
'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Post } from '@/components/Post'
import { useInfiniteQuery } from '@tanstack/react-query'
import { useIntersectionObserver } from '@/hooks/useIntersectionObserver'

interface Post {
  id: string
  author: {
    name: string
    image: string
  }
  content: string
  likes: number
  comments: number
  createdAt: string
}

export function PostFeed() {
  const [ref, inView] = useIntersectionObserver()
  
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage
  } = useInfiniteQuery({
    queryKey: ['posts'],
    queryFn: async ({ pageParam = 1 }) => {
      const res = await fetch(`/api/posts?page=${pageParam}`)
      return res.json()
    },
    getNextPageParam: (lastPage) => lastPage.nextPage
  })
  
  // Load more when bottom is visible
  if (inView && hasNextPage && !isFetchingNextPage) {
    fetchNextPage()
  }
  
  return (
    <div className="space-y-6">
      <AnimatePresence mode="popLayout">
        {data?.pages.map((page) =>
          page.posts.map((post: Post) => (
            <motion.div
              key={post.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              <Post post={post} />
            </motion.div>
          ))
        )}
      </AnimatePresence>
      
      {/* Load more trigger */}
      <div ref={ref} className="h-20">
        {isFetchingNextPage && (
          <div className="flex justify-center">
            <LoadingSpinner />
          </div>
        )}
      </div>
    </div>
  )
}

function LoadingSpinner() {
  return (
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
  )
}