"use client";

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, ChevronRight, BookOpen, MessageSquare, Globe } from 'lucide-react';

interface FAQItem {
    id: number;
    category: string;
    question: string;
    question_ur?: string;
    answer: string;
    answer_ur?: string;
}

interface FaqBrowserProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectQuestion: (question: string) => void;
}

export function FaqBrowser({ isOpen, onClose, onSelectQuestion }: FaqBrowserProps) {
    const [faqs, setFaqs] = useState<FAQItem[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [loading, setLoading] = useState(true);
    const [language, setLanguage] = useState<'en' | 'ur'>('en');

    useEffect(() => {
        if (isOpen) {
            fetch('/api/questions')
                .then(res => res.json())
                .then(data => {
                    setFaqs(data);
                    setLoading(false);
                })
                .catch(err => {
                    console.error('Failed to fetch FAQs:', err);
                    setLoading(false);
                });
        }
    }, [isOpen]);

    const getQuestionText = (faq: FAQItem) => {
        return language === 'ur' ? (faq.question_ur || faq.question) : faq.question;
    };

    const filteredFaqs = faqs.filter(faq => {
        const questionText = getQuestionText(faq);
        return questionText.toLowerCase().includes(searchQuery.toLowerCase()) ||
            faq.category.toLowerCase().includes(searchQuery.toLowerCase());
    });

    const categories = Array.from(new Set(faqs.map(faq => faq.category)));

    const toggleLanguage = () => {
        setLanguage(prev => prev === 'en' ? 'ur' : 'en');
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-background/80 backdrop-blur-sm z-[60]"
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="fixed inset-4 md:inset-x-[20%] md:inset-y-[15%] bg-surface-1 border border-border rounded-2xl md:rounded-3xl z-[70] shadow-2xl flex flex-col overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-4 md:p-6 border-b border-border flex items-center justify-between bg-surface-2/50">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-gold/10 flex items-center justify-center">
                                    <BookOpen className="w-4 h-4 md:w-5 md:h-5 text-gold" />
                                </div>
                                <div>
                                    <h2 className="text-lg md:text-xl font-bold text-foreground">
                                        {language === 'ur' ? 'معلومات کا خزانہ' : 'Knowledge Base'}
                                    </h2>
                                    <p className="text-xs md:text-sm text-muted-foreground">
                                        {language === 'ur' ? 'تمام دستیاب عنوانات اور سوالات براؤز کریں' : 'Browse all available topics and questions'}
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={toggleLanguage}
                                    className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-2 hover:bg-surface-elevated border border-border transition-colors text-xs md:text-sm font-medium"
                                >
                                    <Globe className="w-3.5 h-3.5 md:w-4 md:h-4 text-gold" />
                                    <span>{language === 'en' ? 'English' : 'اردو'}</span>
                                </button>
                                <button
                                    onClick={onClose}
                                    className="p-1.5 md:p-2 hover:bg-surface-elevated rounded-full transition-colors"
                                >
                                    <X className="w-5 h-5 md:w-6 md:h-6 text-muted-foreground" />
                                </button>
                            </div>
                        </div>

                        {/* Search */}
                        <div className="p-4 md:p-6 bg-surface-1">
                            <div className="relative">
                                <Search className={`absolute ${language === 'ur' ? 'right-3.5 md:right-4' : 'left-3.5 md:left-4'} top-1/2 -translate-y-1/2 w-4 h-4 md:w-5 md:h-5 text-muted-foreground`} />
                                <input
                                    type="text"
                                    placeholder={language === 'ur' ? 'سوالات یا زمرے تلاش کریں...' : "Search questions or categories..."}
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className={`w-full bg-surface-2 border border-border rounded-xl md:rounded-2xl ${language === 'ur' ? 'pr-10 md:pr-12 pl-4 text-right' : 'pl-10 md:pl-12 pr-4'} py-2 md:py-3 text-sm md:text-base text-foreground focus:outline-none focus:ring-2 focus:ring-gold/30 transition-all`}
                                    dir={language === 'ur' ? 'rtl' : 'ltr'}
                                />
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 pt-0">
                            {loading ? (
                                <div className="flex items-center justify-center h-full">
                                    <div className="w-8 h-8 border-4 border-gold/30 border-t-gold rounded-full animate-spin" />
                                </div>
                            ) : filteredFaqs.length > 0 ? (
                                <div className="space-y-8" dir={language === 'ur' ? 'rtl' : 'ltr'}>
                                    {categories.map(category => {
                                        const categoryFaqs = filteredFaqs.filter(f => f.category === category);
                                        if (categoryFaqs.length === 0) return null;

                                        return (
                                            <div key={category} className="space-y-3 md:space-y-4">
                                                <h3 className={`text-[10px] md:text-xs font-bold uppercase tracking-widest text-gold/80 px-2 ${language === 'ur' ? 'font-urdu' : ''}`}>
                                                    {category}
                                                </h3>
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-3">
                                                    {categoryFaqs.map(faq => (
                                                        <button
                                                            key={faq.id}
                                                            onClick={() => {
                                                                onSelectQuestion(getQuestionText(faq));
                                                                onClose();
                                                            }}
                                                            className={`flex items-center justify-between p-3 md:p-4 bg-surface-2 hover:bg-surface-elevated border border-border/50 rounded-xl md:rounded-2xl text-left transition-all group ${language === 'ur' ? 'flex-row-reverse text-right' : ''}`}
                                                        >
                                                            <div className={`flex items-center gap-2 md:gap-3 ${language === 'ur' ? 'flex-row-reverse' : ''}`}>
                                                                <MessageSquare className="w-3.5 h-3.5 md:w-4 md:h-4 text-muted-foreground group-hover:text-gold transition-colors shrink-0" />
                                                                <span className={`text-xs md:text-sm font-medium text-foreground/80 group-hover:text-foreground ${language === 'ur' ? 'font-urdu' : ''}`}>
                                                                    {getQuestionText(faq)}
                                                                </span>
                                                            </div>
                                                            <ChevronRight className={`w-3.5 h-3.5 md:w-4 md:h-4 text-muted-foreground group-hover:text-gold transform transition-all shrink-0 ${language === 'ur' ? 'rotate-180 -translate-x-1' : 'group-hover:translate-x-1'}`} />
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-50">
                                    <Search className="w-12 h-12 text-muted-foreground" />
                                    <p className="text-sm text-muted-foreground">
                                        {language === 'ur' ? 'آپ کی تلاش سے مماثل کوئی سوال نہیں ملا۔' : 'No questions found matching your search.'}
                                    </p>
                                </div>
                            )}
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
