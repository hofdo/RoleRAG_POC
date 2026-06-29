import { Component, inject } from '@angular/core';
import { SessionStore } from '../session-store';

@Component({
  selector: 'app-memory-panel',
  template: `
    <section class="memory-panel">
      <header class="memory-panel__header">
        <h2 class="section-label">Memory</h2>
        <button
          type="button"
          class="ghost"
          (click)="store.refreshMemories()"
          [disabled]="!store.sessionId()"
        >
          Refresh
        </button>
      </header>

      @if (store.sessionId()) {
        @if (store.memoryLines().length) {
          <ul class="memory-panel__list">
            @for (line of store.memoryLines(); track $index) {
              <li class="memory-panel__item">{{ line }}</li>
            }
          </ul>
        } @else {
          <p class="memory-panel__empty">No memories yet.</p>
        }
      } @else {
        <p class="memory-panel__hint">Start a session to see memory.</p>
      }
    </section>
  `,
  styles: [`
    /* Grimoire Console styling */
    .memory-panel {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-card);
      padding: 0.9rem;
      color: var(--text);
    }
    .memory-panel__header { display: flex; align-items: center; justify-content: space-between; }
    .memory-panel__header h2 { margin: 0; }
    .memory-panel__list { margin: 0; padding-left: 1.25rem; color: var(--text); }
    .memory-panel__item { margin: 0.25rem 0; }
    .memory-panel__empty, .memory-panel__hint { margin: 0; color: var(--muted); font-style: italic; }
  `],
})
export class MemoryPanelComponent {
  readonly store = inject(SessionStore);
}
