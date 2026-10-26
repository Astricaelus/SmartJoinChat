# Smart Join Chat

Smart Join Chat adds a context-aware active-reply method to AstrBot. Instead of
rolling a fixed probability, a separate model reads recent group messages and
decides whether the bot should join the conversation.

## Features

- Adds **Smart context (LLM)** to `Profile -> Extensions -> Active Reply Method`.
- Shows all Smart Join Chat settings directly below the method selector.
- Uses a dedicated, ephemeral context for the decision call. It does not read or
  mutate the main Agent conversation and does not persist a conversation.
- Resolves the bot's display name from the platform adapter or active persona,
  so calls such as “Are you there, `<bot name>`?” are understood as directed at
  the bot.
- Includes the selected persona prompt as identity/style context for the
  decision, without reusing the main Agent conversation.
- Reuses AstrBot's group-chat context formatter and in-memory history, including
  image captions and the configured maximum history size.
- Collects messages for the decision even when group-context injection is off,
  without injecting those messages into the main Agent.
- Supports low, medium, and high activeness instructions.
- Supports an optional probability gate before the LLM call:
  - **Raw RNG value** uses independent random draws.
  - **Uniform spacing correction** uses a randomized phase and spreads allowed
    messages evenly, reducing bursts of LLM calls.
- Restores the patched AstrBot integration when the plugin is disabled or
  reloaded.

## Installation

Install the repository as an AstrBot plugin and enable it in the WebUI.

```bash
cd AstrBot/data/plugins
git clone https://github.com/Astricaelus/SmartJoinChat.git
```

Restart AstrBot or reload the plugin from `Extensions -> Plugins`.

The plugin targets AstrBot `>=4.29.0,<4.30`.

## Configuration

Open a configuration profile and go to:

`Extensions -> Group Chat Context Awareness -> Active Reply`

Enable active reply, choose **Smart context (LLM)**, and configure:

| Setting | Description |
| --- | --- |
| Decision Model | Chat provider used only for the participation decision. Empty uses the bot's current chat provider. |
| Recent Messages | Number of group messages included in the decision prompt. |
| Activeness | Low, medium, or high participation guidance embedded in the prompt. |
| Probability Gate | Whether to pre-filter messages before calling the decision model. |
| Gate Probability | Probability that a message reaches the decision model. |
| Gate Sampling | Raw independent draws or evenly-spaced opportunities with a randomized phase. |

The normal active-reply whitelist still applies.

## How it works

Smart Join Chat hooks the existing AstrBot active-reply decision point. For the
new method it:

1. Applies the same group-only, wake-message, and whitelist checks as AstrBot.
2. Optionally applies the probability gate.
3. Reads the latest entries from AstrBot's group-context history and formats the
   current message with AstrBot's existing formatter.
4. Calls the selected provider with a separate system prompt, transcript, and
   unique session ID. The selected persona prompt is included as background
   identity/style context, not as conversation history.
5. Accepts only a strict `{"join": boolean, "reason": "..."}` decision. Invalid
   or ambiguous output is treated as "do not join" so a model failure cannot
   create message spam.

The main Agent still writes any actual reply. The decision call never writes to
the main conversation history.

The bot may join based on conversational relevance even when nobody mentions or
invites it by name. A name reference is treated as context to evaluate, not an
automatic reply trigger.

## Compatibility note

AstrBot 4.29.x does not expose a public registry for active-reply strategies or
for adding fields to the profile configuration UI. This plugin therefore adds
the method and fields at runtime through a narrow compatibility layer and
restores the original AstrBot methods when unloaded.

Profile values are mirrored into the plugin's own configuration after a profile
save so they survive an AstrBot process restart. The mirror is internal and does
not affect the main Agent context.

If AstrBot later provides a first-class active-reply strategy registry, the
integration layer can be replaced without changing the decision engine.

## Privacy

The recent group transcript and selected persona prompt are sent to the decision
provider selected in the profile. With an empty provider, it uses the bot's
current chat provider. Review your provider and privacy requirements before
enabling the feature.
