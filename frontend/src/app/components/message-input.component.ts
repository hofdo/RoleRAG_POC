import { Component, computed, inject, signal } from '@angular/core';
import { SessionStore } from '../session-store';

// Grimoire Console styling
@Component({
  selector: 'app-message-input',
  imports: [],
  template: `
    <section class="composer">
      @if (store.busy() && store.currentStage(); as stage) {
        <p class="stage">{{ stageLabel(stage) }}…</p>
      }

      <textarea
        #ta
        class="draft"
        rows="3"
        placeholder="Type your turn…"
        [value]="draft()"
        (input)="draft.set(ta.value)"
      ></textarea>

      <div class="controls">
        <button type="button" class="send" (click)="send()" [disabled]="sendDisabled()">
          Send
        </button>
        <button
          type="button"
          class="ghost"
          (click)="store.rerollLast()"
          [disabled]="store.busy() || !store.sessionId()"
        >
          Reroll last
        </button>
      </div>

      @if (store.turnError(); as error) {
        <p class="error">{{ error }}</p>
      }
    </section>
  `,
  styles: [
    `
      .composer { display: flex; flex-direction: column; gap: 0.5rem; max-width: 720px; }
      .draft { width: 100%; box-sizing: border-box; font: inherit; padding: 0.5rem; }
      .controls { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
      .send:disabled { opacity: 0.5; cursor: not-allowed; }
      .error { color: var(--danger); margin: 0; }
      .stage { color: var(--muted); font-size: 0.8rem; margin: 0; }
    `,
  ],
})
export class MessageInputComponent {
  readonly store = inject(SessionStore);

  readonly draft = signal('');

  readonly sendDisabled = computed(
    () => this.store.busy() || !this.store.sessionId() || this.draft().trim().length === 0,
  );

  private static readonly STAGE_LABELS: Record<string, string> = {
    session: 'Loading session',
    retrieval: 'Retrieving memories',
    routing: 'Choosing route',
    generation: 'Drafting reply',
    validation: 'Checking draft',
    critique: 'Critic reviewing',
    repair: 'Repairing draft',
    persistence: 'Saving turn',
    memory: 'Updating memory',
  };

  stageLabel(stage: string): string {
    return MessageInputComponent.STAGE_LABELS[stage] ?? stage;
  }

  async send(): Promise<void> {
    const ok = await this.store.sendMessage(this.draft().trim());
    if (ok) this.draft.set('');
  }
}
