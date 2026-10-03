const screenHeight = 844;
let screenWidth = 390;
const stage = document.querySelector('.preview-space');
const frame = document.querySelector('.phone-frame');
const screen = document.querySelector('.phone-screen');
const sizeOutput = document.querySelector('#screen-size');
const widthButtons = [...document.querySelectorAll('[data-width]')];

// The iframe keeps its selected CSS viewport size while its preview fits the window.
function fitPreview() {
  const style = getComputedStyle(stage);
  const availableWidth = stage.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const availableHeight = stage.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  const scale = Math.max(0.1, Math.min(1, availableWidth / screenWidth, availableHeight / screenHeight));
  screen.width = screenWidth;
  screen.style.width = `${screenWidth}px`;
  screen.style.transform = `scale(${scale})`;
  frame.style.width = `${screenWidth * scale}px`;
  frame.style.height = `${screenHeight * scale}px`;
}

for (const button of widthButtons) {
  button.addEventListener('click', () => {
    screenWidth = Number(button.dataset.width);
    for (const option of widthButtons) option.setAttribute('aria-pressed', String(option === button));
    sizeOutput.textContent = `${screenWidth} × ${screenHeight}`;
    screen.title = `mooody at ${screenWidth} by ${screenHeight} phone screen size`;
    fitPreview();
  });
}

if (location.hash) {
  screen.src = `./index.html${location.hash}`;
  document.querySelector('.full-app').href = `./index.html${location.hash}`;
}
new ResizeObserver(fitPreview).observe(stage);
fitPreview();
