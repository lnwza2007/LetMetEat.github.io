const imgEl = document.getElementById('Paint');  // Image element (hidden)
const btn = document.getElementById('bth');
const textEl = document.getElementById('text');  // Text element to display random food name

// Create an array of objects to pair text with image paths
const foodData = [
    { text: 'ข้าวซอยไก่', image: 'assets/pic/khao_soi.jpg' },
    { text: 'ข้าวกะเพราไก่', image: 'assets/pic/krapao_gai.jpg' },
    { text: 'ทงคัตสึ', image: 'assets/pic/tonkatsu.jpg' },
    { text: 'แกงกะหรี่ไก่', image: 'assets/pic/chicken_curry.jpg' },
    { text: 'ข้าวมันไก่', image: 'assets/pic/khao_man_gai.jpg' },
    { text: 'พะแนงไก่', image: 'assets/pic/panang_gai.jpg' },
    { text: 'ส้มตำ', image: 'assets/pic/somtam.jpg' },
    { text: 'ข้าวขาหมู', image: 'assets/pic/khaokhamoo.jpg' },
    { text: 'กะเพราหมูกรอบ', image: 'assets/pic/khapaomookob.jpg' },
    { text: 'โจ๊กหมู', image: 'assets/pic/jokmoo.jpg' },
    { text: 'ข้าวต้มหมู', image: 'assets/pic/khaotommoo.jpg' },
    { text: 'เกาเหลาเลือดหมู', image: 'assets/pic/kaohaomoo.jpg' },
    { text: 'เพนเน่คาโบนารา', image: 'assets/pic/Penne-alla-Carbonara.jpg' },
    { text: 'ต้มยำกุ้ง', image: 'assets/pic/tomyumkung.jpg' },
    { text: 'ผัดไทย', image: 'assets/pic/padthai.jpg' },
    { text: 'ข้าวไก่เทอริยากิ', image: 'assets/pic/terriyaki.jpg' },
    { text: 'ไก่ทอด', image: 'assets/pic/kaitod.jpg' },
    { text: 'ก๋วยจั๊บ', image: 'assets/pic/khudjub.jpg' }

];

function randomContent() {
    const randomIndex = Math.floor(Math.random() * foodData.length);  // สุ่มจากarray
    const randomFood = foodData[randomIndex];  // Get random food object

    // โชว์รูป
    textEl.textContent = randomFood.text;  // แสดงชื่อ
    imgEl.src = randomFood.image;  // แสดงรูป
    imgEl.style.display = 'block';  
}

btn.addEventListener('click', () => {
    randomContent(); 
});
