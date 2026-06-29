import { Component, signal, inject } from '@angular/core';
import { SessionStore } from '../session-store';

// Side panel to view/add/delete canon facts for the active session.
@Component({
  selector: 'app-canon-panel',
  imports: [],
  template: `
    <section class="canon">
      <h3 class="section-label">Canon</h3>

      @if (store.canonFacts().length === 0) {
        <p class="empty">No canon facts.</p>
      } @else {
        <ul class="facts">
          @for (fact of store.canonFacts(); track fact.id) {
            <li class="fact">
              <span class="text">{{ fact.text }}</span>
              <button
                type="button"
                class="del ghost"
                title="Delete fact"
                (click)="remove(fact.id)"
              >&#10005;</button>
            </li>
          }
        </ul>
      }

      <div class="add">
        <input
          #factInput
          type="text"
          placeholder="New canon fact…"
          [value]="newFact()"
          (input)="newFact.set(factInput.value)"
        />
        <button
          type="button"
          (click)="add()"
          [disabled]="!store.sessionId() || newFact().trim().length === 0"
        >Add</button>
      </div>
    </section>
  `,
  // Grimoire Console styling
  styles: [`
    .canon {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-card);
      box-shadow: var(--shadow-card);
      padding: 0.9rem;
      color: var(--text);
      font-family: var(--font-body);
    }
    .empty { color: var(--muted); font-style: italic; }
    .facts { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
    .fact {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.5rem;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-chip);
    }
    .text { flex: 1; color: var(--text); }
    .del {
      cursor: pointer;
      background: transparent;
      border: none;
      color: var(--muted);
      padding: 0 0.25rem;
      line-height: 1;
      transition: color 0.15s ease;
    }
    .del:hover { color: var(--danger); }
    .add { display: flex; gap: 0.25rem; }
    .add input {
      flex: 1;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-chip);
      color: var(--text);
      font-family: var(--font-body);
      padding: 0.35rem 0.5rem;
    }
    .add input::placeholder { color: var(--muted); }
    .add input:focus { outline: none; border-color: var(--accent); }
    .add button {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-chip);
      color: var(--accent);
      font-family: var(--font-mono);
      padding: 0.35rem 0.7rem;
      cursor: pointer;
    }
    .add button:hover:not(:disabled) { border-color: var(--accent); }
    .add button:disabled { color: var(--muted); cursor: not-allowed; opacity: 0.6; }
  `],
})
export class CanonPanelComponent {
  readonly store = inject(SessionStore);
  readonly newFact = signal('');

  add(): void {
    const text = this.newFact().trim();
    if (!this.store.sessionId() || text.length === 0) return;
    void this.store.addCanon(text);
    this.newFact.set('');
  }

  remove(factId: string): void {
    void this.store.deleteCanon(factId);
  }
}
