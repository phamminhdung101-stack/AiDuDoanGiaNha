/**
 * HousePrice AI - Core UI Controller
 * Dark Modern AI Platform
 */
document.addEventListener('DOMContentLoaded', () => {
  // 1. Flash message auto-dismiss
  const flashEls = document.querySelectorAll('.flash');
  flashEls.forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(40px)';
    }, 4500);
    setTimeout(() => el.remove(), 5000);
  });

  // 2. Desktop Sidebar collapse / expand toggle
  const sidebar = document.getElementById('appSidebar');
  const appMain = document.getElementById('appMain');
  const collapseBtn = document.getElementById('sidebarCollapseBtn');
  
  if (sidebar && collapseBtn) {
    const isCollapsed = localStorage.getItem('houseai_sidebar_collapsed') === 'true';
    if (isCollapsed) {
      sidebar.classList.add('collapsed');
      if (appMain) appMain.classList.add('collapsed');
    }

    collapseBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      if (appMain) appMain.classList.toggle('collapsed');
      localStorage.setItem('houseai_sidebar_collapsed', sidebar.classList.contains('collapsed'));
    });
  }

  // 3. Mobile Sidebar Drawer toggle
  const mobileToggleBtn = document.getElementById('mobileSidebarToggle');
  const sidebarOverlay = document.getElementById('sidebarOverlay');

  function openMobileSidebar() {
    if (sidebar) sidebar.classList.add('mobile-open');
    if (sidebarOverlay) sidebarOverlay.classList.add('active');
  }

  function closeMobileSidebar() {
    if (sidebar) sidebar.classList.remove('mobile-open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
  }

  if (mobileToggleBtn) {
    mobileToggleBtn.addEventListener('click', openMobileSidebar);
  }
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeMobileSidebar);
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMobileSidebar();
  });
});

/**
 * Toggle password field visibility
 */
function togglePw(btn) {
  const inputWrap = btn.closest('.input-wrap');
  if (!inputWrap) return;
  const input = inputWrap.querySelector('input');
  const icon = btn.querySelector('i');
  if (!input || !icon) return;

  if (input.type === 'password') {
    input.type = 'text';
    icon.classList.replace('ph-eye', 'ph-eye-slash');
  } else {
    input.type = 'password';
    icon.classList.replace('ph-eye-slash', 'ph-eye');
  }
}
