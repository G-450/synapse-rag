import { Pool, neonConfig } from '@neondatabase/serverless';
import { PrismaNeon } from '@prisma/adapter-neon';
import { PrismaClient } from '../generated/prisma/client/client';
import ws from 'ws';

// Setup WebSocket constructor for Neon serverless driver
neonConfig.webSocketConstructor = ws;

// Singleton pattern to avoid multiple Prisma instances in development
const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  throw new Error(
    'DATABASE_URL is not set. Please check your .env file.'
  );
}

// In Prisma 7, PrismaNeon is an adapter factory that takes a PoolConfig object
const adapter = new PrismaNeon({ connectionString });

export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({ adapter });

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma;
