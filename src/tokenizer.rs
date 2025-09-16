//! Simple tokenizer for PicoTron inference demo

use anyhow::Result;
use std::collections::HashMap;

/// Simple character-level tokenizer for demonstration
pub struct SimpleTokenizer {
    vocab_size: usize,
    char_to_id: HashMap<char, u32>,
    id_to_char: HashMap<u32, char>,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a vocabulary
    pub fn new(vocab_size: usize) -> Self {
        // Create a simple character vocabulary
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        
        // Add special tokens (using single characters for simplicity)
        char_to_id.insert('\0', 0);  // PAD
        char_to_id.insert('?', 1);   // UNK
        char_to_id.insert('^', 2);   // START
        char_to_id.insert('$', 3);   // END
        
        id_to_char.insert(0, '\0');
        id_to_char.insert(1, '?');
        id_to_char.insert(2, '^');
        id_to_char.insert(3, '$');
        
        // Add common characters
        let mut id = 4;
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}".chars() {
            if id < vocab_size as u32 {
                char_to_id.insert(c, id);
                id_to_char.insert(id, c);
                id += 1;
            }
        }
        
        Self {
            vocab_size,
            char_to_id,
            id_to_char,
        }
    }
    
    /// Tokenize text into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![2]; // Start token
        
        for c in text.chars() {
            if let Some(&token_id) = self.char_to_id.get(&c) {
                tokens.push(token_id);
            } else {
                tokens.push(1); // Unknown token
            }
        }
        
        tokens.push(3); // End token
        tokens
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        
        for &token_id in tokens {
            if let Some(&c) = self.id_to_char.get(&token_id) {
                match c {
                    '\0' | '^' | '$' => {
                        // Skip special tokens in output
                    }
                    '?' => {
                        text.push('?');
                    }
                    _ => {
                        text.push(c);
                    }
                }
            }
        }
        
        text
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
