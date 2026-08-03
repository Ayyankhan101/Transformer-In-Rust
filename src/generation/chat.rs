use crate::tokenizer::CodeGenTokenizer;

/// Role of a message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Manages multi-turn conversation state and prompt assembly.
pub struct ChatSession {
    history: Vec<Message>,
    system_prompt: Option<String>,
    max_context_tokens: usize,
}

impl ChatSession {
    /// Create a new chat session with a context window limit.
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            history: Vec::new(),
            system_prompt: None,
            max_context_tokens,
        }
    }

    /// Set the system prompt (instruction that applies to all turns).
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    /// Add a user message.
    pub fn add_user(&mut self, content: String) {
        self.history.push(Message {
            role: Role::User,
            content,
        });
    }

    /// Add an assistant message.
    pub fn add_assistant(&mut self, content: String) {
        self.history.push(Message {
            role: Role::Assistant,
            content,
        });
    }

    /// Clear all history (keep system prompt).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get the full conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Assemble the full prompt string from history using Chat format.
    ///
    /// Format:
    /// ```text
    /// <|system|>
    /// {system_prompt}
    /// <|user|>
    /// {turn_1}
    /// <|assistant|>
    /// {response_1}
    /// <|user|>
    /// {current_turn}
    /// <|assistant|>
    /// ```
    pub fn assemble_prompt(&self, current_input: &str) -> String {
        let mut prompt = String::new();

        // System prompt
        if let Some(sys) = &self.system_prompt {
            prompt.push_str("<|system|>\n");
            prompt.push_str(sys);
            prompt.push('\n');
        }

        // History turns
        for msg in &self.history {
            match msg.role {
                Role::System => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                Role::User => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                Role::Assistant => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
            }
        }

        // Current user turn + assistant header (model generates here)
        prompt.push_str("<|user|>\n");
        prompt.push_str(current_input);
        prompt.push('\n');
        prompt.push_str("<|assistant|>\n");

        prompt
    }

    /// Trim oldest turns to fit within the token budget.
    /// Keeps the most recent turns and always preserves the system prompt.
    pub fn trim_to_fit(&mut self, tokenizer: &CodeGenTokenizer) {
        // Build a prompt with all history and check token count
        let full_prompt = self.assemble_prompt("");
        let token_count = tokenizer
            .encode(&full_prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);

        if token_count <= self.max_context_tokens {
            return;
        }

        // Remove oldest turns until we fit (keep at least the last user turn)
        while self.history.len() > 2 {
            let test_prompt = self.assemble_prompt("");
            let count = tokenizer
                .encode(&test_prompt)
                .map(|ids| ids.len())
                .unwrap_or(0);
            if count <= self.max_context_tokens {
                break;
            }
            // Remove the oldest pair (user + assistant)
            self.history.remove(0);
            if !self.history.is_empty() && self.history[0].role == Role::Assistant {
                self.history.remove(0);
            }
        }
    }

    /// Get a formatted history string for display.
    pub fn format_history(&self) -> String {
        let mut output = String::new();

        if let Some(sys) = &self.system_prompt {
            output.push_str(&format!("[System] {sys}\n\n"));
        }

        for msg in &self.history {
            let (label, color_start, color_end) = match msg.role {
                Role::System => ("System", "\x1b[33m", "\x1b[0m"),
                Role::User => ("You", "\x1b[34m", "\x1b[0m"),
                Role::Assistant => ("Assistant", "\x1b[32m", "\x1b[0m"),
            };
            output.push_str(&format!(
                "{color_start}{label}:{color_end} {}\n\n",
                msg.content
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_session_assemble_single_turn() {
        let mut session = ChatSession::new(1024);
        session.add_user("def fibonacci(n):".to_string());

        let prompt = session.assemble_prompt("def fibonacci(n):");
        assert!(prompt.contains("<|user|>\ndef fibonacci(n):\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn chat_session_with_system_prompt() {
        let mut session = ChatSession::new(1024);
        session.set_system_prompt("You are a code assistant.".to_string());
        session.add_user("Write hello world".to_string());

        let prompt = session.assemble_prompt("Write hello world");
        assert!(prompt.starts_with("<|system|>\nYou are a code assistant.\n"));
    }

    #[test]
    fn chat_session_multi_turn() {
        let mut session = ChatSession::new(1024);
        session.add_user("Turn 1".to_string());
        session.add_assistant("Response 1".to_string());

        let prompt = session.assemble_prompt("Turn 2");
        assert!(prompt.contains("<|user|>\nTurn 1\n"));
        assert!(prompt.contains("<|assistant|>\nResponse 1\n"));
        // assemble_prompt adds the current input as a new user turn
        let turns = prompt.matches("<|user|>").count();
        assert_eq!(turns, 2); // Turn 1 from history + Turn 2 from assemble_prompt
    }

    #[test]
    fn chat_session_clear() {
        let mut session = ChatSession::new(1024);
        session.set_system_prompt("sys".to_string());
        session.add_user("hello".to_string());
        session.clear();

        assert!(session.history().is_empty());
        assert_eq!(session.system_prompt(), Some("sys"));
    }

    #[test]
    fn chat_session_format_history() {
        let mut session = ChatSession::new(1024);
        session.set_system_prompt("Be helpful.".to_string());
        session.add_user("Hi".to_string());
        session.add_assistant("Hello!".to_string());

        let formatted = session.format_history();
        // Output contains ANSI color codes around labels
        assert!(formatted.contains("[System] Be helpful."));
        assert!(formatted.contains("You:"));
        assert!(formatted.contains("Hi"));
        assert!(formatted.contains("Assistant:"));
        assert!(formatted.contains("Hello!"));
    }
}
