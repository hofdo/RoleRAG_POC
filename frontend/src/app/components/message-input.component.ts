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
      @if (store.pendingConfirm(); as confirm) {
        <div class="confirm-banner">
          <p>Cloud route requested for this turn.</p>
          <div class="confirm-actions">
            <button type="button" (click)="store.confirmCloud()" [disabled]="store.busy()">
              Confirm cloud
            </button>
            <button type="button" class="ghost" (click)="store.forceLocal()" [disabled]="store.busy()">
              Force local
            </button>
          </div>
        </div>
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
        <label class="cloud-toggle">
          <input
            #cb
            type="checkbox"
            [checked]="requestCloud()"
            (change)="requestCloud.set(cb.checked)"
          />
          Request cloud
        </label>
        <button type="button" class="send" (click)="send()" [disabled]="sendDisabled()">
          Send
        </button>
        <button
          type="button"
          class="ghost"
          (click)="store.rerollLast()"
          [disabled]="store.busy() || !store.sessionId() || !!store.pendingConfirm()"
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
      .cloud-toggle { display: inline-flex; align-items: center; gap: 0.35rem; }
      .send:disabled { opacity: 0.5; cursor: not-allowed; }
      .confirm-banner { border: 1px solid var(--accent); background: var(--surface-2); color: var(--text); padding: 0.5rem 0.75rem; border-radius: var(--r-chip); }
      .confirm-banner p { margin: 0 0 0.5rem; }
      .confirm-actions { display: flex; gap: 0.5rem; }
      .error { color: var(--danger); margin: 0; }
      .stage { color: var(--muted); font-size: 0.8rem; margin: 0; }
    `,
  ],
})
export class MessageInputComponent {
  readonly store = inject(SessionStore);

  readonly draft = signal('');
  readonly requestCloud = signal(false);

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
    const ok = await this.store.sendMessage(this.draft().trim(), this.requestCloud());
    if (ok) this.draft.set('');
  }
}
