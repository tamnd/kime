//! Laya's `build_sequence`: one sequence per question, laid out as
//!
//! ```text
//! [CLS] head [SEP] [MASK] option ... [MASK] option [SEP] state [SEP]
//! ```
//!
//! with the budgets from spec/06-input.md. It takes text that is already rendered (see
//! `kime_core::render`), so this crate stays free of request types.

use crate::Tokenizer;

/// Tokens each option keeps before the crowding rule, as in Laya.
pub const OPTION_CAP: usize = 48;

/// The budgets of a compat checkpoint, from its `rl_agent_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompatBudget {
    /// The whole sequence, 512 for Laya English and 1,024 for multilingual.
    pub max_len: usize,
    /// Head plus options, 192 and 256.
    pub head_max_len: usize,
}

/// Where the state is cut when it does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cut {
    /// Keep the start, Laya's default.
    #[default]
    Tail,
    /// Keep the end, which suits conversations.
    Head,
}

/// One laid out sequence.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompatSequence {
    /// Token ids, at most `max_len` of them.
    pub ids: Vec<u32>,
    /// The position of each option's mask token. Markers past `max_len` are dropped, so fewer
    /// markers than options means the question did not fit.
    pub markers: Vec<u32>,
    /// State tokens before the cut.
    pub state_tokens: usize,
    /// State tokens that made it into the sequence.
    pub state_tokens_used: usize,
}

impl Tokenizer {
    /// Encode the state once, for reuse across every question of a request.
    pub fn encode_state(&self, state: &str) -> Vec<u32> {
        self.encode(state)
    }

    /// Lay out one question. `head` and `options` come from `kime_core::render::compat_question`
    /// and `state_ids` from [`Tokenizer::encode_state`].
    pub fn compat_sequence(
        &self,
        head: &str,
        options: &[String],
        state_ids: &[u32],
        budget: CompatBudget,
        cut: Cut,
    ) -> CompatSequence {
        let sp = self.specials();
        let mut head_ids = self.encode(head);
        let mut opt_ids: Vec<Vec<u32>> = options
            .iter()
            .map(|o| {
                let mut v = Vec::with_capacity(OPTION_CAP + 1);
                v.push(sp.mask);
                self.encode_into(o, &mut v);
                v.truncate(OPTION_CAP + 1);
                v
            })
            .collect();
        // Laya slices the option's own tokens to 48 and then puts the mask in front, so an option
        // holds up to 49 ids here. The crowding rule below cuts the whole thing, mask included.
        let total = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>();
        let mut opt_budget = budget.head_max_len as isize - total(&opt_ids) as isize;
        if opt_budget < 16 {
            let per = ((budget.head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
            for o in &mut opt_ids {
                o.truncate(per);
            }
            opt_budget = budget.head_max_len as isize - total(&opt_ids) as isize;
        }
        head_ids.truncate(opt_budget.max(8) as usize);

        let mut ids = Vec::with_capacity(budget.max_len);
        ids.push(sp.cls);
        ids.extend_from_slice(&head_ids);
        ids.push(sp.sep);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len() as u32);
            ids.extend_from_slice(o);
        }
        ids.push(sp.sep);
        let prefix = ids.len();
        let room = budget.max_len.saturating_sub(prefix + 1);
        let used = room.min(state_ids.len());
        match cut {
            Cut::Tail => ids.extend_from_slice(&state_ids[..used]),
            Cut::Head => ids.extend_from_slice(&state_ids[state_ids.len() - used..]),
        }
        ids.push(sp.sep);
        ids.truncate(budget.max_len);
        markers.retain(|&m| (m as usize) < budget.max_len);
        // When the head and options alone overflow, the final cut can take state tokens too.
        let used = used.min(ids.len().saturating_sub(prefix));
        CompatSequence { ids, markers, state_tokens: state_ids.len(), state_tokens_used: used }
    }
}
