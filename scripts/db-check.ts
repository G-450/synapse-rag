import 'dotenv/config';
import { prisma } from '../src/lib/prisma';

async function run() {
  try {
    console.log("Checking tables...");
    const res = await prisma.$queryRawUnsafe(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public';
    `);
    console.log("Tables in public schema:", res);
  } catch (e) {
    console.error(e);
  } finally {
    await prisma.$disconnect();
  }
}

run();
