// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-side-projects",
          title: "side projects",
          description: "Things I&#39;ve worked on in my spare time, for fun or for learning",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Professional history and CV",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-the-principal-engineer-paradigm",
        
          title: "The Principal Engineer Paradigm",
        
        description: "Understanding the Role",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/principal_engineers_paradigm/";
          
        },
      },{id: "post-80-20-mlops",
        
          title: "80/20 MLOps",
        
        description: "A Basic MLOps Setup With Kubeflow",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/80_20_MLOPS/";
          
        },
      },{id: "post-charming-algorithms-1-39-3sum-closest-39",
        
          title: "Charming Algorithms 1 - &#39;3Sum Closest&#39;",
        
        description: "Intuition For the solution to &#39;3Sum - Closest&#39;",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/charming-algorithms-1-3Sum-Closest/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-planyourmeals-com-2017-2020",
          title: 'PlanYourMeals.com 2017 - 2020',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects";
            },},{id: "projects-dronedog",
          title: 'DroneDog',
          description: "VLMs and navigation for home security",
          section: "Projects",handler: () => {
              window.location.href = "/projects";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6A%61%6D%65%73.%64%76%61%6E%63%65@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/jamesdvance", "_blank");
        },
      },{
        id: 'social-kaggle',
        title: 'Kaggle',
        section: 'Socials',
        handler: () => {
          window.open("https://www.kaggle.com/jamesdvance", "_blank");
        },
      },{
        id: 'social-leetcode',
        title: 'LeetCode',
        section: 'Socials',
        handler: () => {
          window.open("https://leetcode.com/u/jamesdvance/", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/jamesdvance", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
