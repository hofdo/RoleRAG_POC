import { Component, inject, OnInit } from '@angular/core';
import { SessionStore } from '../session-store';
import { SetupPickerComponent } from '../components/setup-picker.component';
import { TranscriptComponent } from '../components/transcript.component';
import { MessageInputComponent } from '../components/message-input.component';
import { MemoryPanelComponent } from '../components/memory-panel.component';
import { CanonPanelComponent } from '../components/canon-panel.component';

// Phase 1 play loop: status banner + setup picker + transcript + composer, with memory and
// canon side panels. Each leaf component injects SessionStore directly — this container only
// arranges them and kicks off the initial status/catalog loads.
@Component({
  selector: 'app-play',
  imports: [
    SetupPickerComponent,
    TranscriptComponent,
    MessageInputComponent,
    MemoryPanelComponent,
    CanonPanelComponent,
  ],
  template: `
    <section class="play">
      <div class="play__main">
        @if (store.statusView().warning) {
          <p class="warn">{{ store.statusView().warning }}</p>
        }
        <app-setup-picker />
        <app-transcript />
        <app-message-input />
      </div>
      <aside class="play__aside">
        <app-memory-panel />
        <app-canon-panel />
      </aside>
    </section>
  `,
  styles: [
    `
      .play { display: grid; grid-template-columns: minmax(0, 1fr) 18rem; gap: 1.5rem; align-items: start; }
      .play__main { display: flex; flex-direction: column; gap: 1rem; min-width: 0; }
      .play__aside { display: flex; flex-direction: column; gap: 1.5rem; }
      .warn { color: var(--danger); margin: 0; }
      @media (max-width: 800px) {
        .play { grid-template-columns: 1fr; }
      }
    `,
  ],
})
export class PlayComponent implements OnInit {
  readonly store = inject(SessionStore);

  ngOnInit(): void {
    void this.store.loadRuntimeStatus();
    void this.store.loadCatalog();
  }
}
