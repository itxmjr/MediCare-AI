
import { NextResponse } from 'next/server';
import questions from '@/data/questions.json';

export async function GET() {
    return NextResponse.json(questions);
}
